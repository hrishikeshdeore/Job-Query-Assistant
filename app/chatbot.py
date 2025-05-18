import json
import os
import requests # For making HTTP requests to the Inference API

# Determine the base directory for data files consistently
# This assumes chatbot.py is in the 'app' directory, and 'data' is a subdirectory of 'app'.
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # app directory
DATA_DIR = os.path.join(BASE_DIR, "data")

# Revert to flan-t5-base as originally intended
MODEL_API_URL = "https://api-inference.huggingface.co/models/meta-llama/Llama-3.1-8B-Instruct"

class JobChatBot:
    def __init__(self, hf_api_token=None):
        """
        Initialize the Job ChatBot to use Hugging Face Inference API.
        """
        self.hf_api_token = hf_api_token
        if not self.hf_api_token:
            # This print is mostly for server-side logging if needed, Streamlit UI shows errors too.
            print("CRITICAL_CHATBOT_INIT: Hugging Face API Token not provided to JobChatBot constructor.")

        self.conversation_history = []
        self.load_data()
        self.model = True # Placeholder
        self.tokenizer = True # Placeholder
        
    def load_data(self):
        """
        Load FAQs, mock jobs, and cities data from JSON files.
        """
        try:
            with open(os.path.join(DATA_DIR, 'faqs.json'), 'r', encoding='utf-8') as f:
                self.faqs = json.load(f)['faqs']
            with open(os.path.join(DATA_DIR, 'mock_jobs.json'), 'r', encoding='utf-8') as f:
                self.jobs = json.load(f)['jobs']
            with open(os.path.join(DATA_DIR, 'cities.json'), 'r', encoding='utf-8') as f:
                self.cities = json.load(f)['cities']
        except FileNotFoundError as e:
            print(f"Error loading data file: {e}. Ensure data files are in {DATA_DIR}")
            self.faqs, self.jobs, self.cities = [], [], []
        except Exception as e:
            print(f"An unexpected error occurred loading data: {e}")
            self.faqs, self.jobs, self.cities = [], [], []
    
    def get_system_prompt_and_context(self, user_query):
        """
        Generate the prompt for the T5 model, including relevant context (FAQs, job data).
        Modified to prevent the model from generating a simulated conversation.
        """
        # System instructions
        prompt_start = """You are JobSevak, a helpful job assistant for Lokal's Jobs platform in India.
Your task is to respond to the user's query in a helpful, concise manner.
IMPORTANT: Only generate ONE RESPONSE as JobSevak. Do not create a simulated conversation.
DO NOT generate any text that appears to be from the user. Only respond as the assistant.

"""
        
        # Add context about cities
        cities_text = ", ".join(self.cities)
        prompt_start += f"Available cities for job search: {cities_text}.\n\n"
        
        # Add FAQs
        faq_text = "\n".join([f"Q: {faq['question']}\nA: {faq['answer']}" for faq in self.faqs])
        prompt_start += f"Reference FAQs:\n{faq_text}\n\n"

        # Add job data context if relevant
        job_data_context = self.get_job_data_context(user_query)
        if job_data_context:
            prompt_start += f"Relevant job listings for this query:\n{job_data_context}\n\n"

        # Add conversation history
        if self.conversation_history:
            history_text = ""
            for turn in self.conversation_history[-4:]:  # Last 2 exchanges
                role = "User" if turn["role"] == "user" else "JobSevak"
                history_text += f"{role}: {turn['content']}\n"
            
            prompt_start += f"Previous conversation:\n{history_text}\n"

        # Add the current query with explicit instruction to only respond as JobSevak
        final_prompt = f"{prompt_start}Current query - User: {user_query}\n\nYour response (respond ONLY as JobSevak):"
        return final_prompt

    def query_job_listings(self, location=None, job_type=None, gender_preference=None):
        """
        Filter job listings based on location, job type or gender preference.
        """
        filtered_jobs = self.jobs
        if location:
            filtered_jobs = [job for job in filtered_jobs if job['location'].lower() == location.lower()]
        if job_type:
            filtered_jobs = [job for job in filtered_jobs if job_type.lower() in job['job_type'].lower()]
        if gender_preference:
            filtered_jobs = [job for job in filtered_jobs if job['gender_preference'].lower() == gender_preference.lower() or job['gender_preference'].lower() == 'any']
        return filtered_jobs
    
    def get_job_data_context(self, query):
        """
        Analyze the query to see if we need to provide job listings as context.
        Returns relevant job data as context string if needed.
        """
        job_related_keywords = ['job', 'work', 'employment', 'career', 'listing', 'opportunity']
        is_job_query = any(keyword in query.lower() for keyword in job_related_keywords)
        if not is_job_query: return ""
        
        location = next((city for city in self.cities if city.lower() in query.lower()), None)
        
        gender_preference = None
        if 'women' in query.lower() or 'female' in query.lower(): gender_preference = 'Female'
        elif 'men' in query.lower() or 'male' in query.lower(): gender_preference = 'Male'
        
        job_type = None
        if 'part-time' in query.lower() or 'part time' in query.lower(): job_type = 'Part-time'
        elif 'full-time' in query.lower() or 'full time' in query.lower(): job_type = 'Full-time'
        
        filtered_jobs = self.query_job_listings(location, job_type, gender_preference)
        if not filtered_jobs and location: return f"No jobs found in {location} matching your current criteria."
        if not (location or job_type or gender_preference) and is_job_query: filtered_jobs = self.jobs[:3]
        
        if filtered_jobs:
            job_context = ""
            for job in filtered_jobs[:3]:
                job_context += f"- Title: {job['title']}, Co: {job['company']}, Loc: {job['location']}, Sal: {job['salary']}\n"
            return job_context
        return ""
    
    def chat(self, user_message):
        """
        Process user message and generate response using Hugging Face Inference API.
        """
        if not self.hf_api_token:
            return "I apologize, but the connection to the language model service is not configured. HF_API_TOKEN is missing. Please ensure it is set in your .env file and restart the application."

        self.conversation_history.append({"role": "user", "content": user_message})
        if len(self.conversation_history) > 6: 
            self.conversation_history = self.conversation_history[-6:]
        
        prompt_for_api = self.get_system_prompt_and_context(user_message)
        
        headers = {"Authorization": f"Bearer {self.hf_api_token}"}
        payload = {
            "inputs": prompt_for_api,
            "parameters": {
                "max_new_tokens": 150,
                "num_return_sequences": 1,
                "temperature": 0.7,
                "top_p": 0.95,
                "do_sample": True,
                "no_repeat_ngram_size": 2,
                "early_stopping": True
            },
            "options": {
                "wait_for_model": True # If the model is not ready, wait for it to load
            }
        }
        
        try:
            api_response = requests.post(MODEL_API_URL, headers=headers, json=payload, timeout=30)
            api_response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
            
            result = api_response.json()
            response_text = ""
            
            if isinstance(result, list) and result:
                response_text = result[0].get('generated_text', '')
                
                # If the model included prompt in response, remove it
                if response_text.startswith(prompt_for_api):
                    response_text = response_text[len(prompt_for_api):].strip()
                
                # Additional processing to find and extract only the assistant's response
                assistant_marker = "JobSevak:"
                user_marker = "User:"
                
                # If the response has "JobSevak:" markers, extract just the first part after that
                if assistant_marker in response_text:
                    parts = response_text.split(assistant_marker)
                    if len(parts) > 1:
                        # Get text after first "JobSevak:" but before any "User:" if present
                        assistant_part = parts[1].strip()
                        if user_marker in assistant_part:
                            assistant_part = assistant_part.split(user_marker)[0].strip()
                        response_text = assistant_part
                
                # Clean up any additional JobSevak: prefixes
                if response_text.startswith(assistant_marker):
                    response_text = response_text[len(assistant_marker):].strip()
                    
            elif isinstance(result, dict) and 'generated_text' in result:
                response_text = result.get('generated_text', '')
                
                # Apply the same cleanup logic
                assistant_marker = "JobSevak:"
                user_marker = "User:"
                
                if assistant_marker in response_text:
                    parts = response_text.split(assistant_marker)
                    if len(parts) > 1:
                        assistant_part = parts[1].strip()
                        if user_marker in assistant_part:
                            assistant_part = assistant_part.split(user_marker)[0].strip()
                        response_text = assistant_part
                        
                if response_text.startswith(assistant_marker):
                    response_text = response_text[len(assistant_marker):].strip()
                    
            elif isinstance(result, dict) and 'error' in result:
                response_text = f"Model API Error: {result['error']}"
                if 'estimated_time' in result:
                    response_text += f" The model might be loading, estimated time: {result['estimated_time']:.2f}s."
            else:
                response_text = "Sorry, I received an unexpected response from the model service."
                print(f"Unexpected API response format: {result}")

            # Ensure we don't have any residual "JobSevak:" prefixes or "User:" segments
            clean_response = response_text.strip()
            
            self.conversation_history.append({"role": "assistant", "content": clean_response})
            return clean_response
            
        except requests.exceptions.RequestException as e:
            error_message = f"Network issue connecting to model API: {str(e)}. Please try again later."
            print(error_message)
            self.conversation_history.append({"role": "assistant", "content": error_message})
            return error_message
        except Exception as e:
            error_message = f"Unexpected error during API call: {str(e)}."
            print(error_message)
            self.conversation_history.append({"role": "assistant", "content": error_message})
            return error_message 