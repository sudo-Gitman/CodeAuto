
![DALL·E 2025-01-28 23 06 21 - A retro-style GitHub repository banner for 'CodeAuto'  The design features an 80s aesthetic with vibrant neon colors, a grid background, and glowing t](https://github.com/user-attachments/assets/36fca34b-68d7-4a37-9789-c15e3fddc5eb)


# AutoGPT with Modern GUI

A powerful AutoGPT implementation with a modern graphical user interface built using CustomTkinter. This application combines the power of large language models with an intuitive interface for enhanced developer productivity.

## Features

- 🤖 Advanced AI-powered code generation and optimization
- 🎨 Modern, user-friendly GUI with dark/light mode support
- 📊 Real-time performance monitoring and analytics
- 🔌 Plugin system for extensibility
- 📁 Code management and export capabilities
- 📈 Task queuing and execution tracking
- 🔄 Asynchronous operation for better performance
- 🖥️ System resource monitoring
- 🔒 Secure plugin management

## Prerequisites

- Python 3.9 or higher
- Docker (optional, for containerized deployment)
- NVIDIA GPU (optional, for improved performance)

## Installation

### Option 1: Local Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/sudo-Gitman/CodeAuto.git
   cd CodeAuto-project
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   
   # On Windows
   .\venv\Scripts\activate
   
   # On Linux/Mac
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Option 2: Docker Installation

1. Clone the repository:
   ```bash
  
   git clone https://github.com/sudo-Gitman/CodeAuto.git
   cd autogpt-project
   ```

2. Build and run using Docker Compose:
   ```bash
   docker-compose up --build
   ```

## Running the Application

### Local Run

1. Activate the virtual environment if not already activated:
   ```bash
   # On Windows
   .\venv\Scripts\activate
   
   # On Linux/Mac
   source venv/bin/activate
   ```

2. Start the application:
   ```bash
   python gui.py
   ```

### Docker Run

1. Start the application:
   ```bash
   docker-compose up
   ```

2. For subsequent runs:
   ```bash
   docker-compose start
   ```

## Using the Application

1. **Main Interface**
   - Use the sidebar to navigate between different views
   - Chat view for interacting with AutoGPT
   - Analytics view for performance monitoring
   - Code view for managing generated code
   - Settings for configuration

2. **Working with Goals**
   - Enter your development goal in the chat input
   - Click "Send" or press Enter to submit
   - Monitor progress in real-time
   - View generated code and suggestions

3. **Managing Code**
   - Generated code is automatically saved
   - Use the Code view to browse files
   - Export code using the Export button
   - Share code directly from the interface

4. **Monitoring Performance**
   - View real-time system metrics
   - Track task success rates
   - Monitor resource usage
   - Analyze execution times

5. **Plugin Management**
   - Install plugins from the Settings view
   - Enable/disable plugins as needed
   - Configure plugin settings
   - View plugin status

## Running Tests

### Local Testing

1. Install test dependencies:
   ```bash
   pip install pytest pytest-asyncio pytest-cov
   ```

2. Run tests:
   ```bash
   python -m pytest tests/ -v
   ```

### Docker Testing

1. Make the test script executable:
   ```bash
   chmod +x run_tests.sh
   ```

2. Run tests in Docker:
   ```bash
   ./run_tests.sh
   ```

## Project Structure

```
CodeAuto-project/
├── autogpt.py           # Core AutoGPT implementation
├── gui.py              # GUI implementation
├── requirements.txt    # Python dependencies
├── Dockerfile         # Main Dockerfile
├── Dockerfile.test    # Testing Dockerfile
├── docker-compose.yml # Docker Compose configuration
├── run_tests.sh      # Test runner script
├── generated_code/   # Generated code storage
├── plugins/         # Plugin directory
└── logs/           # Application logs
```

## Troubleshooting

1. **GUI Not Showing in Docker**
   - Ensure X11 forwarding is properly configured
   - Check DISPLAY environment variable
   - Verify host network mode is enabled

2. **Plugin Installation Issues**
   - Verify plugin compatibility
   - Check plugin directory permissions
   - Review plugin logs for errors

3. **Performance Issues**
   - Monitor system resources
   - Adjust worker count in settings
   - Consider enabling GPU support

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- CustomTkinter for the modern GUI framework
- Transformers library for AI capabilities
- The open-source community for various tools and libraries
