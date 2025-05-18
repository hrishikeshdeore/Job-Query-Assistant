# JobSevak - Lokal Job Assistant
## https://jobask.streamlit.app/



![image](https://github.com/user-attachments/assets/0285aada-c7f4-45cb-86f5-90f36c500bd3)


# JobSevak - Lokal Job Assistant

A conversational AI chatbot designed to help users find job opportunities in Tier 2/3 Indian cities through Lokal's Jobs Vertical. Built as a demonstration of LLM integration and conversational AI capabilities, this project showcases various aspects of modern chatbot development.

## Key Components & Features
- Interactive chat interface using Streamlit
- LLM-powered responses using Hugging Face's Inference API
- Support for text (voice support is in progress)
- Context-aware conversations
- Job recommendations for various Indian cities
- FAQ support for common job-related queries

### LLM Integration (25%)
- Integration with Hugging Face's Inference API using `meta-llama/Llama-3.1-8B-Instruct` model to use/host without downlading large models locally
- Carefully structured prompts for job-related queries
- Context-aware prompt engineering for natural conversations
- Efficient API usage with proper error handling

### Core Functionality (25%)
- Job search with location, type, and preference filters
- FAQ handling for common job-related queries
- Multi-turn conversation support
- Structured responses for job listings and information
- Mock data integration for demonstration purposes

### User Experience (15%)
- Clean, intuitive Streamlit-based web interface
- Voice input support using WebRTC is in progress not shown in ui
- Real-time response streaming (no need to download large models locally)
- Clear conversation history display
- Error handling with user-friendly messages

### Code Quality (15%)
- Modular architecture separating concerns:
  - `app.py`: UI and user interaction
  - `chatbot.py`: LLM integration and business logic
  - `data/`: Structured mock data storage
- Clean, documented code following Python best practices
- Environment variable management for secure API key handling
- Comprehensive error handling

### Scalability & Extensibility (10%)
- Easy integration of new job sources
- Modular design for adding new features
- Configurable LLM model selection
- Extensible mock data structure

### Bonus Features (10%)
- Voice input support using Whisper API will be available soon
- Multilingual
- Multi-turn conversation context
- Structured data integration
- Real-time response generation



## Setup Instructions

1. Clone the repository:
```bash
git clone <your-repo-url>
cd job_search_chatbot
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure environment:
   - Create a `.env` file in the root directory
   - Add your Hugging Face API token:
```
HF_API_TOKEN=your_hugging_face_api_token_here
```

4. Run the application:
```bash
streamlit run app/app.py
```

## Project Structure

```
job_search_chatbot/
├── app/
│   ├── app.py          # Streamlit UI and main application logic
│   ├── chatbot.py      # LLM integration and conversation handling
│   └── data/           # Mock data for jobs, FAQs, and cities
├── requirements.txt    # Project dependencies
├── .env               # Environment variables (not in repo)
└── README.md          # Project documentation
```

## Technical Details

### LLM Integration
- Model: `meta-llama/Llama-3.1-8B-Instruct` via Hugging Face Inference API
- Prompt Structure:
  - Context injection for job data
  - Conversation history maintenance
  - Response formatting guidelines
  - Supports multiple languages without coontext

### Data Management
- Mock data stored in JSON format
- Easily extensible data structure
- Support for multiple data types:
  - Job listings
  - FAQs
  - City information
  - User preferences

### Error Handling
- API failure recovery
- Invalid input management
- Rate limiting consideration
- User feedback for all error states

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
