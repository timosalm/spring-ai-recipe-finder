# Prerequisites and Installation Requirements

This document outlines the prerequisites and installation requirements for running the Spring AI Recipe Finder application.

## System Requirements

### Hardware Requirements
- **Minimum RAM**: 8GB (recommended for running with local LLM via Ollama)
- **Disk Space**: At least 2GB free space for dependencies and models
- **CPU**: Multi-core processor recommended for optimal performance

### Software Requirements

#### 1. Java Development Kit (JDK)
- **Required Version**: Java 21 (LTS)
- **Recommended Distribution**: Eclipse Temurin (formerly AdoptOpenJDK)

**Installation Options:**

##### Option A: Using SDKMAN! (Recommended for Linux/macOS)
```bash
# Install SDKMAN!
curl -s "https://get.sdkman.io" | bash
source "$HOME/.sdkman/bin/sdkman-init.sh"

# Install Java 21
sdk install java 21.0.2-tem
sdk use java 21.0.2-tem

# Verify installation
java -version
```

##### Option B: Direct Download
- **Download**: [Eclipse Temurin JDK 21](https://adoptium.net/temurin/releases/?version=21)
- **Verify**: `java -version` should show version 21.x.x

##### Option C: Package Managers

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install -y openjdk-21-jdk
java -version
```

**macOS (Homebrew):**
```bash
brew install openjdk@21
# Add to PATH
echo 'export PATH="/opt/homebrew/opt/openjdk@21/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
java -version
```

**Windows (Chocolatey):**
```powershell
choco install temurin21
java -version
```

#### 2. Gradle (Optional - Wrapper Included)
The project includes Gradle Wrapper, so you **don't need to install Gradle manually**. The wrapper will automatically download Gradle 8.8.

**Using Gradle Wrapper:**
```bash
# Linux/macOS
./gradlew --version

# Windows
gradlew.bat --version
```

**If you prefer to install Gradle system-wide:**

##### Using SDKMAN! (Linux/macOS)
```bash
sdk install gradle 8.8
gradle --version
```

##### Using Homebrew (macOS)
```bash
brew install gradle
gradle --version
```

##### Using Chocolatey (Windows)
```bash
choco install gradle
gradle --version
```

#### 3. Docker and Docker Compose (Optional but Recommended)
Required for running Redis vector store and Ollama (if using local LLM).

**Installation:**
- **Download**: [Docker Desktop](https://www.docker.com/products/docker-desktop)
- **Verify**: 
  ```bash
  docker --version
  docker compose version
  ```

**Linux Alternative (Docker Engine):**
```bash
# Install Docker Engine
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Install Docker Compose plugin
sudo apt-get update
sudo apt-get install docker-compose-plugin

# Add user to docker group
sudo usermod -aG docker $USER
newgrp docker
```

#### 4. Git
Required for cloning the repository.

```bash
# Verify
git --version

# Install if needed
# Ubuntu/Debian: sudo apt install git
# macOS: brew install git
# Windows: choco install git
```

## Quick Setup Verification

After installing the prerequisites, verify your setup:

```bash
# Check Java version (should be 21)
java -version

# Check Gradle (via wrapper)
./gradlew --version

# Check Docker (if using)
docker --version
docker compose version

# Check Git
git --version
```

Expected output example:
```
java -version
openjdk version "21.0.2" 2024-01-16 LTS
OpenJDK Runtime Environment Temurin-21.0.2+13 (build 21.0.2+13-LTS)
OpenJDK 64-Bit Server VM Temurin-21.0.2+13 (build 21.0.2+13-LTS, mixed mode)

./gradlew --version
Gradle 8.8

docker --version
Docker version 24.0.7, build afdd53b

git --version
git version 2.43.0
```

## Installation Steps

### 1. Clone the Repository
```bash
git clone https://github.com/timosalm/spring-ai-recipe-finder.git
cd spring-ai-recipe-finder
```

### 2. Verify Build
```bash
# This will download all dependencies and compile the project
./gradlew build -x test

# Expected output: BUILD SUCCESSFUL
```

### 3. Run the Application
```bash
# Start the application
./gradlew bootRun

# The application will start on http://localhost:8080
```

## Additional Setup for Different LLM Providers

### Option 1: Local LLM with Ollama

**Install Ollama:**
```bash
# Linux
curl -fsSL https://ollama.com/install.sh | sh

# macOS
brew install ollama

# Windows: Download from https://ollama.com/download
```

**Start Ollama and download model:**
```bash
# Start Ollama service (Linux/macOS)
ollama serve

# In another terminal, pull the model
ollama pull llama3.2

# Verify
ollama list
```

**Run the application:**
```bash
./gradlew bootRun
# Application will automatically connect to local Ollama
```

### Option 2: OpenAI

**Set up API key:**
```bash
# Set environment variable
export SPRING_AI_OPENAI_API_KEY=sk-your-api-key-here

# Run with OpenAI profile
export SPRING_PROFILES_ACTIVE=openai
./gradlew bootRun
```

### Option 3: Azure OpenAI

**Set up credentials:**
```bash
export SPRING_AI_AZURE_OPENAI_API_KEY=your-api-key
export SPRING_AI_AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com

# Run with Azure profile
export SPRING_PROFILES_ACTIVE=azure
./gradlew bootRun
```

## IDE Setup (Optional)

### IntelliJ IDEA
1. Install IntelliJ IDEA (Community or Ultimate)
2. Open the project folder
3. IDE will automatically detect Gradle and configure the project
4. Wait for dependency download to complete
5. Run the application using the Spring Boot run configuration

### VS Code
1. Install VS Code
2. Install extensions:
   - Extension Pack for Java
   - Spring Boot Extension Pack
   - Gradle for Java
3. Open the project folder
4. Use the Spring Boot Dashboard to run the application

### Eclipse
1. Install Eclipse IDE for Enterprise Java Developers
2. Install Spring Tools 4 (Help → Eclipse Marketplace → Search "Spring Tools 4")
3. Import as Gradle project
4. Run as Spring Boot App

## Docker Setup (Alternative)

If you prefer to run the application in a container:

```bash
# Build the container image
./gradlew bootBuildImage --imageName=recipe-finder

# Run the container
docker run -p 8080:8080 recipe-finder
```

## Development Container (VS Code)

The project includes a `.devcontainer` configuration for VS Code:

1. Install VS Code and the "Dev Containers" extension
2. Open the project folder in VS Code
3. Click "Reopen in Container" when prompted
4. The container will automatically set up Java 21 and Gradle

## Troubleshooting

### Issue: "JAVA_HOME is not set"
```bash
# Find Java installation
which java

# Set JAVA_HOME (Linux/macOS)
export JAVA_HOME=$(/usr/libexec/java_home -v 21)  # macOS
export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64  # Linux

# Add to ~/.bashrc or ~/.zshrc to make permanent
```

### Issue: "Permission denied: ./gradlew"
```bash
# Make gradlew executable
chmod +x gradlew
./gradlew --version
```

### Issue: Gradle download is slow
```bash
# Use a different Gradle distribution mirror
# Edit gradle/wrapper/gradle-wrapper.properties
# Change distributionUrl to use a closer mirror
```

### Issue: OutOfMemoryError during build
```bash
# Increase Gradle memory
export GRADLE_OPTS="-Xmx2048m"
./gradlew build
```

### Issue: Docker is not running
```bash
# Start Docker Desktop (Windows/macOS)
# Or start Docker daemon (Linux)
sudo systemctl start docker
```

## Minimum Working Setup

For a minimal working setup, you only need:
1. **Java 21** - Required
2. **Git** - To clone the repository
3. The **Gradle Wrapper** is included (no separate installation needed)

The application will work with the included Gradle wrapper and no additional tools, though Docker is recommended for the Redis vector store.

## Next Steps

After completing the setup:
1. Read [README.md](../README.md) for usage instructions
2. Review [SECURITY.md](SECURITY.md) for security configuration
3. Check [SECURITY_EXAMPLES.md](SECURITY_EXAMPLES.md) for practical examples
4. Run the tests: `./gradlew test`
5. Access the application at http://localhost:8080

## Support

For issues or questions:
- Check existing [GitHub Issues](https://github.com/timosalm/spring-ai-recipe-finder/issues)
- Review the [Spring AI Documentation](https://docs.spring.io/spring-ai/reference/)
- Consult the [Spring Boot Documentation](https://docs.spring.io/spring-boot/docs/current/reference/html/)
