# AI-Powered Sentiment Analysis API

This project is a **FastAPI-based microservice** that leverages **Hugging Face's Transformers** to perform sentiment analysis on text input, such as customer reviews. The API is containerized using **Docker** and can be deployed to **AWS (ECS or Lambda)** for scalability. It’s a practical demonstration of integrating AI models with a modern backend framework, perfect for showcasing skills in AI, backend development, and cloud deployment.

## Features
- **Sentiment Analysis**: Analyzes text to determine positive, negative, or neutral sentiment using a pre-trained Hugging Face model.
- **FastAPI Backend**: Provides a fast, asynchronous API for real-time sentiment analysis.
- **Dockerized**: Packaged in a Docker container for easy deployment and scalability.
- **Scalable & Production-Ready**: Built with best practices for microservices and AI integration.

## Tech Stack
- **Backend**: FastAPI (Python)
- **AI/ML**: Hugging Face Transformers
- **Containerization**: Docker
- **Dependencies**: Python 3.9+, PyTorch, Transformers, Uvicorn

## Getting Started

### Prerequisites
- Python 3.9 or higher
- Docker (for containerization)
- Git

### Installation
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/Ali-Ismail-1/sentiment-analysis-api
   cd sentiment-analysis-api
   ```

2. **Set Up a Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the API Locally**:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000
   ```

5. **Access the API**:
   - Open your browser or use a tool like Postman to visit `http://localhost:8000/docs` for the interactive API documentation.
   - Test the endpoint `/analyze` with a POST request (see Usage below).

### Docker Setup
1. **Build the Docker Image**:
   ```bash
   docker build -t sentiment-analysis-api .
   ```

2. **Run the Docker Container**:
   ```bash
   docker run -p 8000:8000 sentiment-analysis-api
   ```

3. **Access the API**:
   - Visit `http://localhost:8000/docs` to interact with the API.

### Example Usage
- **Endpoint**: `POST /analyze`
- **Request Body**:
  ```json
  {
    "text": "I absolutely loved the service at this restaurant!"
  }
  ```
- **Response**:
  ```json
  {
    "text": "I absolutely loved the service at this restaurant!",
    "sentiment": "positive",
    "confidence": 0.98
  }
  ```


## Project Structure
```
sentiment-analysis-api/
├── main.py                # FastAPI application
├── requirements.txt       # Python dependencies
├── Dockerfile             # Docker configuration
├── README.md              # This file
```

## Future Improvements
- Add support for multiple languages using multilingual models.
- Implement batch processing for analyzing multiple texts.
- Add authentication for secure API access.
- Enhance error handling and logging for production use.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request with your improvements.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

## Contact
Feel free to reach out via [GitHub Issues](https://github.com/Ali-Ismail-1/sentiment-analysis-api/issues) or connect with me on [LinkedIn](https://www.linkedin.com/in/ali-ismail-35196615/) or [X](https://x.com/Ali_F_Ismail) for feedback or questions!

---

Built with ❤️ using FastAPI, Hugging Face, and Docker.