import customtkinter as ctk
import asyncio
import threading
import queue
import time
from datetime import datetime
import json
from pathlib import Path
import markdown2
from PIL import Image
import logging
from typing import Optional, List, Dict, Any
import psutil
import platform
from concurrent.futures import ThreadPoolExecutor

from autogpt import AsyncAutoGPT, Config

# Configure customtkinter appearance
ctk.set_appearance_mode("system")
ctk.set_default_color_theme("blue")

class AsyncTkinter:
    """Helper class to run async code in Tkinter"""
    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()

    def _run_loop(self):
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def run_async(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self.loop)

class SystemMonitor:
    """Monitor system resources"""
    @staticmethod
    def get_system_info() -> Dict[str, Any]:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return {
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "memory_used": f"{memory.used / (1024**3):.1f}GB",
            "memory_total": f"{memory.total / (1024**3):.1f}GB",
            "disk_percent": disk.percent,
            "disk_used": f"{disk.used / (1024**3):.1f}GB",
            "disk_total": f"{disk.total / (1024**3):.1f}GB"
        }

class TaskManager:
    """Manages task execution, queuing and visualization"""
    def __init__(self):
        self.tasks = []
        self.current_task = None
        self.task_history = []
        self.task_metrics = {
            'completed': 0,
            'failed': 0,
            'total_time': 0
        }

    async def add_task(self, goal: str, priority: int = 1):
        """Add a new task to the queue"""
        task = {
            'id': len(self.tasks) + 1,
            'goal': goal,
            'status': 'pending',
            'priority': priority,
            'created_at': datetime.now(),
            'started_at': None,
            'completed_at': None,
            'duration': None,
            'error': None
        }
        self.tasks.append(task)
        return task

    async def execute_next_task(self, autogpt):
        """Execute the next task in queue"""
        if not self.tasks:
            return None
            
        self.current_task = self.tasks.pop(0)
        self.current_task['status'] = 'running'
        self.current_task['started_at'] = datetime.now()
        
        try:
            await autogpt.run_goal(self.current_task['goal'])
            self.current_task['status'] = 'completed'
            self.task_metrics['completed'] += 1
        except Exception as e:
            self.current_task['status'] = 'failed'
            self.current_task['error'] = str(e)
            self.task_metrics['failed'] += 1
            
        self.current_task['completed_at'] = datetime.now()
        self.current_task['duration'] = (
            self.current_task['completed_at'] - 
            self.current_task['started_at']
        ).total_seconds()
        
        self.task_metrics['total_time'] += self.current_task['duration']
        self.task_history.append(self.current_task)
        self.current_task = None
        
        return self.current_task

class CodeManager:
    """Manages code generation, export and sharing"""
    def __init__(self, base_dir: str = "generated_code"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        self.current_session = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_dir = self.base_dir / self.current_session
        self.session_dir.mkdir(exist_ok=True)
        
    def save_code(self, code: str, filename: str, language: str = "python"):
        """Save generated code to file"""
        file_path = self.session_dir / filename
        with open(file_path, 'w') as f:
            f.write(code)
        return file_path
        
    def export_session(self, format: str = "zip") -> Path:
        """Export current session as archive"""
        if format == "zip":
            archive_path = self.base_dir / f"session_{self.current_session}.zip"
            import shutil
            shutil.make_archive(str(archive_path.with_suffix('')), 'zip', self.session_dir)
            return archive_path
            
    def get_session_files(self) -> List[Path]:
        """Get list of files in current session"""
        return list(self.session_dir.glob("*.*"))

class PluginManager:
    """Manages plugins and their lifecycle"""
    def __init__(self):
        self.plugins = {}
        self.hooks = {
            'pre_goal_execution': [],
            'post_goal_execution': [],
            'on_error': [],
            'on_code_generated': []
        }
        
    def register_plugin(self, name: str, plugin: Dict):
        """Register a new plugin"""
        if name in self.plugins:
            raise ValueError(f"Plugin {name} already registered")
            
        required_methods = ['initialize', 'cleanup']
        for method in required_methods:
            if method not in plugin:
                raise ValueError(f"Plugin {name} missing required method: {method}")
                
        self.plugins[name] = plugin
        
        # Register hooks
        for hook_name, hook_fn in plugin.get('hooks', {}).items():
            if hook_name in self.hooks:
                self.hooks[hook_name].append(hook_fn)
                
        # Initialize plugin
        plugin['initialize']()
        
    def unregister_plugin(self, name: str):
        """Unregister a plugin"""
        if name not in self.plugins:
            return
            
        plugin = self.plugins[name]
        
        # Cleanup plugin
        plugin['cleanup']()
        
        # Remove hooks
        for hook_fns in self.hooks.values():
            hook_fns[:] = [fn for fn in hook_fns 
                          if fn.__module__ != plugin.__module__]
                          
        del self.plugins[name]
        
    async def execute_hook(self, hook_name: str, *args, **kwargs):
        """Execute all functions registered for a hook"""
        results = []
        for hook_fn in self.hooks.get(hook_name, []):
            try:
                if asyncio.iscoroutinefunction(hook_fn):
                    result = await hook_fn(*args, **kwargs)
                else:
                    result = hook_fn(*args, **kwargs)
                results.append(result)
            except Exception as e:
                logger.error(f"Error executing hook {hook_name} in {hook_fn}: {e}")
        return results

class ModernAutoGPTGUI(ctk.CTk):
    def __init__(self):
        super().__init__()

        # Initialize async support
        self.async_tk = AsyncTkinter()
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.queue = queue.Queue()
        
        # Initialize AutoGPT
        self.config = Config(
            model_name="gpt2",
            max_workers=3,
            enable_cache=True
        )
        self.autogpt = AsyncAutoGPT(self.config)
        
        # Initialize system monitor
        self.system_monitor = SystemMonitor()
        
        # Initialize task manager
        self.task_manager = TaskManager()
        
        # Initialize code manager
        self.code_manager = CodeManager()
        
        # Initialize plugin manager
        self.plugin_manager = PluginManager()
        
        self.setup_gui()
        self.setup_logging()
        
        # Start monitoring
        self.monitor_system_resources()
        self.process_queue()

    def setup_gui(self):
        """Setup enhanced GUI components"""
        self.title("AutoGPT - AI Development Assistant")
        self.geometry("1400x900")
        
        # Configure grid
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)
        self.grid_columnconfigure(2, weight=0)

        # Create components
        self.setup_sidebar()
        self.setup_main_content()
        self.setup_system_monitor()
        self.setup_status_bar()

    def setup_sidebar(self):
        """Setup enhanced sidebar"""
        sidebar = ctk.CTkFrame(self, width=200, corner_radius=0)
        sidebar.grid(row=0, column=0, rowspan=2, sticky="nsew")
        sidebar.grid_rowconfigure(6, weight=1)

        # Logo and title
        logo_frame = ctk.CTkFrame(sidebar)
        logo_frame.grid(row=0, column=0, padx=20, pady=20, sticky="ew")
        
        title = ctk.CTkLabel(
            logo_frame,
            text="AutoGPT",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        title.grid(row=0, column=0, padx=10)

        # Navigation buttons with icons
        self.nav_buttons = {}
        nav_items = [
            ("💭 Chat", self.show_chat),
            ("📊 History", self.show_history),
            ("📈 Analytics", self.show_analytics),
            ("📁 Code", self.show_code),  # New code view
            ("⚙️ Settings", self.show_settings),
            ("❓ Help", self.show_help)
        ]

        for i, (text, command) in enumerate(nav_items, 1):
            btn = ctk.CTkButton(
                sidebar,
                text=text,
                command=command,
                anchor="w",
                height=40,
                font=ctk.CTkFont(size=14)
            )
            btn.grid(row=i, column=0, padx=20, pady=10, sticky="ew")
            self.nav_buttons[text] = btn

    def setup_main_content(self):
        """Setup enhanced main content area"""
        self.main_content = ctk.CTkFrame(self)
        self.main_content.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)
        self.main_content.grid_rowconfigure(0, weight=1)
        self.main_content.grid_columnconfigure(0, weight=1)

        # Setup all views
        self.setup_chat_view()
        self.setup_history_view()
        self.setup_analytics_view()
        self.setup_code_view()  # New code view
        self.setup_settings_view()
        self.setup_help_view()

        # Show chat by default
        self.show_chat()

    def setup_chat_view(self):
        """Setup enhanced chat interface"""
        self.chat_frame = ctk.CTkFrame(self.main_content)
        self.chat_frame.grid_rowconfigure(0, weight=1)
        self.chat_frame.grid_columnconfigure(0, weight=1)

        # Chat display with syntax highlighting
        self.chat_display = ctk.CTkTextbox(
            self.chat_frame,
            wrap="word",
            font=ctk.CTkFont(family="Consolas", size=12)
        )
        self.chat_display.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="nsew")

        # Input area
        input_frame = ctk.CTkFrame(self.chat_frame)
        input_frame.grid(row=1, column=0, padx=20, pady=(0, 20), sticky="ew")
        input_frame.grid_columnconfigure(0, weight=1)

        self.goal_input = ctk.CTkEntry(
            input_frame,
            placeholder_text="Enter your development goal...",
            height=40,
            font=ctk.CTkFont(size=14)
        )
        self.goal_input.grid(row=0, column=0, padx=(0, 10), sticky="ew")

        send_btn = ctk.CTkButton(
            input_frame,
            text="Send",
            command=self.send_goal,
            width=100,
            height=40,
            font=ctk.CTkFont(size=14)
        )
        send_btn.grid(row=0, column=1)

    def setup_analytics_view(self):
        """Setup enhanced analytics dashboard"""
        self.analytics_frame = ctk.CTkFrame(self.main_content)
        
        # Task metrics
        metrics_frame = ctk.CTkFrame(self.analytics_frame)
        metrics_frame.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        
        self.metrics_labels = {}
        metrics = [
            "Total Tasks", "Success Rate", "Avg Duration",
            "Queue Length", "Cache Hit Rate", "Error Rate"
        ]
        
        for i, metric in enumerate(metrics):
            label = ctk.CTkLabel(
                metrics_frame,
                text=f"{metric}: --",
                font=ctk.CTkFont(size=14)
            )
            label.grid(row=i//2, column=i%2, padx=20, pady=10)
            self.metrics_labels[metric] = label

        # Task visualization
        viz_frame = ctk.CTkFrame(self.analytics_frame)
        viz_frame.grid(row=1, column=0, padx=20, pady=20, sticky="nsew")
        
        # Task history table
        self.task_table = ctk.CTkTextbox(
            viz_frame, 
            wrap="none",
            height=200,
            font=ctk.CTkFont(family="Consolas", size=12)
        )
        self.task_table.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        # Update metrics periodically
        self.update_task_metrics()
        
    def update_task_metrics(self):
        """Update task metrics display"""
        metrics = self.task_manager.task_metrics
        total_tasks = metrics['completed'] + metrics['failed']
        
        if total_tasks > 0:
            success_rate = (metrics['completed'] / total_tasks) * 100
            avg_duration = metrics['total_time'] / total_tasks
            error_rate = (metrics['failed'] / total_tasks) * 100
            
            self.metrics_labels["Total Tasks"].configure(
                text=f"Total Tasks: {total_tasks}"
            )
            self.metrics_labels["Success Rate"].configure(
                text=f"Success Rate: {success_rate:.1f}%"
            )
            self.metrics_labels["Avg Duration"].configure(
                text=f"Avg Duration: {avg_duration:.2f}s"
            )
            self.metrics_labels["Queue Length"].configure(
                text=f"Queue Length: {len(self.task_manager.tasks)}"
            )
            self.metrics_labels["Error Rate"].configure(
                text=f"Error Rate: {error_rate:.1f}%"
            )
            
        # Update task history table
        self.task_table.delete("1.0", "end")
        headers = ["ID", "Goal", "Status", "Duration", "Error"]
        self.task_table.insert("end", " | ".join(headers) + "\n")
        self.task_table.insert("end", "-" * 80 + "\n")
        
        for task in self.task_manager.task_history[-10:]:  # Show last 10 tasks
            row = [
                str(task['id']),
                task['goal'][:30] + "..." if len(task['goal']) > 30 else task['goal'],
                task['status'],
                f"{task['duration']:.1f}s" if task['duration'] else "--",
                task['error'][:30] + "..." if task['error'] and len(task['error']) > 30 else task['error'] or "--"
            ]
            self.task_table.insert("end", " | ".join(row) + "\n")
            
        self.after(2000, self.update_task_metrics)

    def setup_code_view(self):
        """Setup code management view"""
        self.code_frame = ctk.CTkFrame(self.main_content)
        
        # File browser
        browser_frame = ctk.CTkFrame(self.code_frame)
        browser_frame.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        
        # File list
        self.file_list = ctk.CTkTextbox(
            browser_frame,
            wrap="none",
            height=300,
            font=ctk.CTkFont(family="Consolas", size=12)
        )
        self.file_list.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        # Action buttons
        btn_frame = ctk.CTkFrame(browser_frame)
        btn_frame.grid(row=1, column=0, padx=10, pady=10, sticky="ew")
        
        export_btn = ctk.CTkButton(
            btn_frame,
            text="Export Session",
            command=self.export_current_session
        )
        export_btn.grid(row=0, column=0, padx=5, pady=5)
        
        share_btn = ctk.CTkButton(
            btn_frame,
            text="Share Code",
            command=self.share_code
        )
        share_btn.grid(row=0, column=1, padx=5, pady=5)
        
        # Update file list periodically
        self.update_file_list()
        
    def update_file_list(self):
        """Update the file browser"""
        self.file_list.delete("1.0", "end")
        
        files = self.code_manager.get_session_files()
        if not files:
            self.file_list.insert("end", "No files generated yet...")
            return
            
        self.file_list.insert("end", "Generated Files:\n\n")
        for file in files:
            self.file_list.insert("end", f"📄 {file.name}\n")
            
        self.after(5000, self.update_file_list)
        
    def export_current_session(self):
        """Export the current coding session"""
        try:
            archive_path = self.code_manager.export_session()
            self.update_status(
                f"Session exported to: {archive_path}",
                "success"
            )
        except Exception as e:
            self.update_status(f"Export failed: {str(e)}", "error")
            
    def share_code(self):
        """Share code via clipboard or external service"""
        try:
            files = self.code_manager.get_session_files()
            if not files:
                self.update_status("No files to share", "info")
                return
                
            # For now, just copy paths to clipboard
            paths = "\n".join(str(f) for f in files)
            self.clipboard_clear()
            self.clipboard_append(paths)
            
            self.update_status("File paths copied to clipboard", "success")
        except Exception as e:
            self.update_status(f"Share failed: {str(e)}", "error")

    def setup_system_monitor(self):
        """Setup system resource monitor"""
        monitor_frame = ctk.CTkFrame(self, width=250)
        monitor_frame.grid(row=0, column=2, padx=20, pady=20, sticky="nsew")
        
        # Resource usage
        self.cpu_progress = self.create_resource_meter(monitor_frame, "CPU Usage", 0)
        self.memory_progress = self.create_resource_meter(monitor_frame, "Memory Usage", 1)
        self.disk_progress = self.create_resource_meter(monitor_frame, "Disk Usage", 2)

    def create_resource_meter(self, parent, label_text: str, row: int) -> ctk.CTkProgressBar:
        """Create a resource usage meter"""
        frame = ctk.CTkFrame(parent)
        frame.grid(row=row, column=0, padx=10, pady=10, sticky="ew")
        
        label = ctk.CTkLabel(frame, text=label_text)
        label.grid(row=0, column=0, padx=5, pady=5)
        
        progress = ctk.CTkProgressBar(frame)
        progress.grid(row=1, column=0, padx=5, pady=5, sticky="ew")
        progress.set(0)
        
        return progress

    def monitor_system_resources(self):
        """Update system resource monitoring"""
        try:
            info = self.system_monitor.get_system_info()
            
            self.cpu_progress.set(info["cpu_percent"] / 100)
            self.memory_progress.set(info["memory_percent"] / 100)
            self.disk_progress.set(info["disk_percent"] / 100)
            
        except Exception as e:
            logger.error(f"Error monitoring resources: {e}")
        
        finally:
            self.after(2000, self.monitor_system_resources)

    async def process_goal(self, goal: str):
        """Process a goal asynchronously using task manager and plugins"""
        try:
            # Execute pre-goal hooks
            await self.plugin_manager.execute_hook('pre_goal_execution', goal)
            
            self.update_status("Adding task to queue...", "info")
            task = await self.task_manager.add_task(goal)
            
            self.update_status(f"Processing task {task['id']}...", "running")
            result = await self.task_manager.execute_next_task(self.autogpt)
            
            # Execute post-goal hooks
            await self.plugin_manager.execute_hook('post_goal_execution', goal, result)
            
            self.update_status("Task completed", "success")
        except Exception as e:
            # Execute error hooks
            await self.plugin_manager.execute_hook('on_error', goal, e)
            
            self.update_status(f"Error: {str(e)}", "error")
            logger.error(f"Error processing goal: {e}")

    def setup_settings_view(self):
        """Setup enhanced settings view"""
        self.settings_frame = ctk.CTkFrame(self.main_content)
        
        # Create tabs
        tabview = ctk.CTkTabview(self.settings_frame)
        tabview.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        
        # General settings tab
        general_tab = tabview.add("General")
        self.setup_general_settings(general_tab)
        
        # Plugin settings tab
        plugins_tab = tabview.add("Plugins")
        self.setup_plugin_settings(plugins_tab)
        
    def setup_plugin_settings(self, parent):
        """Setup plugin management UI"""
        # Plugin list
        list_frame = ctk.CTkFrame(parent)
        list_frame.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        
        # Plugin list display
        self.plugin_list = ctk.CTkTextbox(
            list_frame,
            wrap="none",
            height=200,
            font=ctk.CTkFont(family="Consolas", size=12)
        )
        self.plugin_list.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        
        # Action buttons
        btn_frame = ctk.CTkFrame(list_frame)
        btn_frame.grid(row=1, column=0, padx=10, pady=10, sticky="ew")
        
        install_btn = ctk.CTkButton(
            btn_frame,
            text="Install Plugin",
            command=self.install_plugin
        )
        install_btn.grid(row=0, column=0, padx=5, pady=5)
        
        remove_btn = ctk.CTkButton(
            btn_frame,
            text="Remove Plugin",
            command=self.remove_plugin
        )
        remove_btn.grid(row=0, column=1, padx=5, pady=5)
        
        # Update plugin list
        self.update_plugin_list()
        
    def update_plugin_list(self):
        """Update the plugin list display"""
        self.plugin_list.delete("1.0", "end")
        
        if not self.plugin_manager.plugins:
            self.plugin_list.insert("end", "No plugins installed...")
            return
            
        self.plugin_list.insert("end", "Installed Plugins:\n\n")
        for name, plugin in self.plugin_manager.plugins.items():
            self.plugin_list.insert("end", f"🔌 {name}\n")
            if 'description' in plugin:
                self.plugin_list.insert("end", f"   {plugin['description']}\n")
            self.plugin_list.insert("end", "\n")
            
    def install_plugin(self):
        """Show dialog to install a new plugin"""
        dialog = ctk.CTkInputDialog(
            text="Enter plugin package name or path:",
            title="Install Plugin"
        )
        plugin_path = dialog.get_input()
        if not plugin_path:
            return
            
        try:
            # Here we would implement actual plugin installation
            # For now, just show a success message
            self.update_status(
                f"Plugin installation not implemented yet",
                "info"
            )
        except Exception as e:
            self.update_status(f"Plugin installation failed: {str(e)}", "error")
            
    def remove_plugin(self):
        """Show dialog to remove a plugin"""
        if not self.plugin_manager.plugins:
            self.update_status("No plugins to remove", "info")
            return
            
        dialog = ctk.CTkInputDialog(
            text="Enter plugin name to remove:",
            title="Remove Plugin"
        )
        plugin_name = dialog.get_input()
        if not plugin_name:
            return
            
        try:
            self.plugin_manager.unregister_plugin(plugin_name)
            self.update_status(f"Plugin {plugin_name} removed", "success")
            self.update_plugin_list()
        except Exception as e:
            self.update_status(f"Plugin removal failed: {str(e)}", "error")

    def send_goal(self):
        """Send goal to AutoGPT"""
        goal = self.goal_input.get()
        if not goal:
            return

        self.goal_input.delete(0, "end")
        self.update_chat_display(f"\nYou: {goal}\n")
        
        # Run async goal processing
        self.async_tk.run_async(self.process_goal(goal))

    def update_status(self, message: str, status: str = "info"):
        """Update status with color coding"""
        colors = {
            "info": "gray",
            "running": "blue",
            "success": "green",
            "error": "red"
        }
        self.status_label.configure(
            text=message,
            text_color=colors.get(status, "gray")
        )

    def _log_performance_metrics(self):
        """Log and display performance metrics"""
        if hasattr(self.autogpt, 'task_times'):
            total_tasks = len(self.autogpt.task_times)
            if total_tasks > 0:
                avg_time = sum(self.autogpt.task_times.values()) / total_tasks
                self.metrics_labels["Total Tasks"].configure(text=f"Total Tasks: {total_tasks}")
                self.metrics_labels["Average Task Time"].configure(text=f"Average Task Time: {avg_time:.2f}s")

def main():
    app = ModernAutoGPTGUI()
    app.mainloop()

if __name__ == "__main__":
    main()
