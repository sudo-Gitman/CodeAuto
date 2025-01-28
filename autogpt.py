import os
import json
import logging
import subprocess
import asyncio
import aiofiles
from typing import List, Tuple, Optional, Dict, Any, Union
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache, partial
from pathlib import Path
from dataclasses import dataclass, asdict, field
import time
import pickle
from contextlib import asynccontextmanager

from transformers import pipeline, GPT2LMHeadModel, GPT2Tokenizer, PreTrainedModel, PreTrainedTokenizer
from langchain import LLMChain, PromptTemplate
from langchain.llms import HuggingFacePipeline
from langchain.cache import InMemoryCache
import torch

# Configure logging with custom formatter
class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors and better timestamp format"""
    COLORS = {
        'DEBUG': '\033[94m',    # Blue
        'INFO': '\033[92m',     # Green
        'WARNING': '\033[93m',  # Yellow
        'ERROR': '\033[91m',    # Red
        'CRITICAL': '\033[91m', # Red
        'RESET': '\033[0m'      # Reset
    }

    def format(self, record):
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        record.msg = f"{color}{record.msg}{self.COLORS['RESET']}"
        return super().format(record)

logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
handler.setFormatter(ColoredFormatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)
logger.setLevel(logging.INFO)

@dataclass
class Config:
    """Enhanced configuration settings for AutoGPT."""
    model_name: str = "gpt2"
    max_workers: int = 3
    max_retries: int = 3
    retry_delay: float = 1.0
    output_dir: str = "output"
    docker_base_image: str = "python:3.9"
    execution_timeout: int = 60
    cache_dir: str = ".cache"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 4
    max_memory_mb: int = 1024
    enable_cache: bool = True
    debug_mode: bool = False

    def __post_init__(self):
        """Ensure directories exist."""
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)

class CacheManager:
    """Manages caching for various components."""
    def __init__(self, config: Config):
        self.config = config
        self.cache_dir = Path(config.cache_dir)
        self.memory_cache = InMemoryCache()
        self.model_cache = {}

    @asynccontextmanager
    async def cached_operation(self, key: str, ttl: int = 3600):
        """Async context manager for caching operations."""
        cache_file = self.cache_dir / f"{key}.pickle"
        
        if cache_file.exists() and time.time() - cache_file.stat().st_mtime < ttl:
            async with aiofiles.open(cache_file, 'rb') as f:
                content = await f.read()
                yield pickle.loads(content)
                return

        result = yield None

        if result is not None:
            async with aiofiles.open(cache_file, 'wb') as f:
                await f.write(pickle.dumps(result))

class ModelManager:
    """Manages AI models with efficient loading and caching."""
    def __init__(self, config: Config):
        self.config = config
        self.cache_manager = CacheManager(config)
        self._models: Dict[str, PreTrainedModel] = {}
        self._tokenizers: Dict[str, PreTrainedTokenizer] = {}

    @lru_cache(maxsize=2)
    def get_model(self, model_name: str) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
        """Get or load model and tokenizer with caching."""
        if model_name not in self._models:
            logger.info(f"Loading model: {model_name}")
            self._tokenizers[model_name] = GPT2Tokenizer.from_pretrained(
                model_name,
                cache_dir=self.config.cache_dir
            )
            self._models[model_name] = GPT2LMHeadModel.from_pretrained(
                model_name,
                cache_dir=self.config.cache_dir
            ).to(self.config.device)

        return self._models[model_name], self._tokenizers[model_name]

    def create_pipeline(self, model_name: str) -> HuggingFacePipeline:
        """Create an optimized pipeline for the model."""
        model, tokenizer = self.get_model(model_name)
        generator = pipeline(
            "text-generation",
            model=model,
            tokenizer=tokenizer,
            device=self.config.device,
            batch_size=self.config.batch_size
        )
        return HuggingFacePipeline(pipeline=generator)

class AsyncFileManager:
    """Asynchronous file operations manager."""
    @staticmethod
    async def read_file(file_path: str) -> Optional[str]:
        """Read file asynchronously."""
        try:
            async with aiofiles.open(file_path, 'r') as f:
                return await f.read()
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {e}")
            return None

    @staticmethod
    async def write_file(file_path: str, content: str) -> bool:
        """Write file asynchronously."""
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            async with aiofiles.open(file_path, 'w') as f:
                await f.write(content)
            return True
        except Exception as e:
            logger.error(f"Error writing to file {file_path}: {e}")
            return False

class AsyncDockerManager:
    """Asynchronous Docker operations manager."""
    def __init__(self, config: Config):
        self.config = config

    async def build_image(self, dockerfile_path: str, image_name: str) -> bool:
        """Build Docker image asynchronously."""
        for attempt in range(self.config.max_retries):
            try:
                process = await asyncio.create_subprocess_exec(
                    "docker", "build", "-t", image_name, dockerfile_path,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                stdout, stderr = await process.communicate()
                
                if process.returncode == 0:
                    logger.info(f"Docker image '{image_name}' built successfully")
                    return True
                
                logger.error(f"Error building Docker image: {stderr.decode()}")
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(self.config.retry_delay)
            except Exception as e:
                logger.error(f"Exception building Docker image: {e}")
                if attempt < self.config.max_retries - 1:
                    await asyncio.sleep(self.config.retry_delay)
        return False

    async def run_container(self, image_name: str, container_name: str) -> Tuple[Optional[str], Optional[str]]:
        """Run Docker container asynchronously."""
        try:
            process = await asyncio.create_subprocess_exec(
                "docker", "run", "--rm", "--name", container_name, image_name,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                return stdout.decode(), None
            return None, stderr.decode()
        except Exception as e:
            return None, str(e)
        finally:
            # Cleanup container
            try:
                await asyncio.create_subprocess_exec(
                    "docker", "rm", "-f", container_name,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL
                )
            except Exception:
                pass

class AsyncAutoGPT:
    """Asynchronous AutoGPT system with improved performance."""
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.model_manager = ModelManager(self.config)
        self.cache_manager = CacheManager(self.config)
        self.file_manager = AsyncFileManager()
        self.docker_manager = AsyncDockerManager(self.config)
        
        # Initialize components
        self.llm = self.model_manager.create_pipeline(self.config.model_name)
        self.task_planner = create_task_planner(self.llm)
        self.task_executor = create_task_executor(self.llm)
        self.code_generator = create_code_generator(self.llm)
        
        # Performance monitoring
        self.start_time = time.time()
        self.task_times: Dict[str, float] = {}

    async def run_goal(self, goal: str) -> None:
        """Run the AutoGPT system asynchronously."""
        logger.info(f"Starting AutoGPT with goal: {goal}")
        self.start_time = time.time()

        try:
            # Plan tasks
            tasks = await self._plan_tasks(goal)
            
            # Execute tasks concurrently
            async with asyncio.TaskGroup() as tg:
                task_coros = [
                    tg.create_task(self._execute_task_with_feedback(task, goal))
                    for task in tasks
                ]

            # Process results
            results = [task.result() for task in task_coros if task.result()]
            
            # Save and analyze results
            await self._save_results(results)
            self._log_performance_metrics()

        except* Exception as e:
            logger.error(f"Error executing goal: {e}")
            raise

    async def _plan_tasks(self, goal: str) -> List[str]:
        """Plan tasks with caching."""
        cache_key = f"task_plan_{hash(goal)}"
        
        async with self.cache_manager.cached_operation(cache_key) as cached:
            if cached:
                return cached

            tasks_response = await asyncio.to_thread(self.task_planner.run, goal=goal)
            tasks = tasks_response.strip().split("\n")
            logger.info(f"Planned Tasks:\n{tasks_response}")
            return tasks

    async def _execute_task_with_feedback(self, task: str, goal: str) -> Optional[str]:
        """Execute a task with performance monitoring."""
        task_start = time.time()
        try:
            result = await self._execute_single_task(task, goal)
            self.task_times[task] = time.time() - task_start
            return result
        except Exception as e:
            logger.error(f"Task failed: {task}\nError: {e}")
            return None

    async def _execute_single_task(self, task: str, goal: str) -> Optional[str]:
        """Execute a single task with optimized resource usage."""
        if "code" in task.lower() or "webpage" in task.lower():
            return await self._handle_code_task_with_feedback(task, goal)
        
        result = await asyncio.to_thread(self.task_executor.run, task=task)
        return result

    def _log_performance_metrics(self):
        """Log performance metrics."""
        total_time = time.time() - self.start_time
        logger.info("\nPerformance Metrics:")
        logger.info(f"Total execution time: {total_time:.2f}s")
        
        if self.task_times:
            avg_task_time = sum(self.task_times.values()) / len(self.task_times)
            logger.info(f"Average task time: {avg_task_time:.2f}s")
            
            slowest_task = max(self.task_times.items(), key=lambda x: x[1])
            logger.info(f"Slowest task: {slowest_task[0]} ({slowest_task[1]:.2f}s)")

        if torch.cuda.is_available():
            logger.info(f"GPU Memory used: {torch.cuda.max_memory_allocated() / 1024**2:.2f}MB")

async def main():
    """Async main function."""
    config = Config()
    autogpt = AsyncAutoGPT(config)
    
    while True:
        try:
            goal = input("\nEnter your goal (or 'exit' to quit): ")
            if goal.lower() == 'exit':
                break
                
            await autogpt.run_goal(goal)
            
        except KeyboardInterrupt:
            logger.info("\nGracefully shutting down...")
            break
        except Exception as e:
            logger.error(f"An error occurred: {e}")

if __name__ == "__main__":
    asyncio.run(main())
