# JobSevak - Lokal Job Assistant
![image](https://github.com/user-attachments/assets/0285aada-c7f4-45cb-86f5-90f36c500bd3)


A conversational AI chatbot designed to help users find job opportunities in Tier 2/3 Indian cities through Lokal's Jobs Vertical. The chatbot provides personalized job recommendations and answers queries about job opportunities, requirements, and application processes.

## Features

- Interactive chat interface using Streamlit
- LLM-powered responses using Hugging Face's Inference API
- Support for both text and voice input
- Context-aware conversations
- Job recommendations for various Indian cities
- FAQ support for common job-related queries

## Setup

1. Clone the repository:
```bash
git clone <your-repo-url>
cd job_search_chatbot
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the root directory and add your Hugging Face API token:
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
│   ├── app.py          # Main Streamlit application
│   ├── chatbot.py      # Chatbot logic and API integration
│   └── data/           # Mock data for jobs, FAQs, etc.
├── requirements.txt    # Project dependencies
└── README.md          # Project documentation
```

## Environment Variables

- `HF_API_TOKEN`: Hugging Face API token for accessing the language model

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details. # Jobsearch-Assistant
