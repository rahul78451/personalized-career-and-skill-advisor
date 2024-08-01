# CareerSage

CareerSage is an AI-powered chatbot designed to provide personalized career advice. It analyzes resumes, suggests job opportunities based on user skills and experiences, and offers valuable interview tips. Leveraging SerpApi, CareerSage fetches and scrapes career-related articles, building a rich knowledge base for delivering expert-backed career guidance. With integration from reputable career advice books and the Blog Chatbot feature, CareerSage ensures comprehensive support for career navigation.

## Use Case

CareerSage has the potential to transform career counseling and job search platforms by:

- **Job Seekers**: Providing resume feedback, job recommendations aligned with skills, and expert interview preparation.
- **Career Counselors and HR Professionals**: Offering insights to effectively guide clients or candidates with well-rounded, up-to-date career advice.

CareerSage delivers accessible, personalized, and data-driven career guidance, helping users navigate their professional journeys with ease.

## Technologies

- **OpenAI’s GPT models**
- **Gemini Model**
- **LangChain**
- **Streamlit**

## Workflow

1. **Generate Vector Embeddings**: Use OpenAI/Gemini embeddings in LangChain.
2. **Store Vector Embeddings**: Save them in the Faiss Vector Database.
3. **Fetch Relevant Chunk**: Retrieve the most relevant chunk of the source based on user input using vector embeddings.
4. **Create Prompt Template**: Design a template for prompts.
5. **Generate Response**: Send user input and relevant chunk to ChatGPT.
6. **Display Chat**: Show the conversation on the Streamlit dashboard.

## Applications

- **Educational Institutions**: Integrate the chatbot into school and university portals to guide students towards their career paths.
- **Corporate Onboarding**: Assist new employees in understanding their career growth and development opportunities within the company.
- **Remote Areas**: Provide career advice in geographically isolated locations where traditional counseling is scarce.
- **Continuous Learning Platforms**: Offer career advice based on skills acquired through online courses or MOOCs.


## Getting Started

1. Clone the repository: `git clone https://github.com/sanikamal/career-advisor-chatbot.git`
2. Install dependencies: `pip install -r requirements.txt`
3. Run the application: `streamlit run app.py`

## Contributing

Contributions are welcome! Please read the [Contributing Guidelines]() before making a pull request.

## License

This project is licensed under the MIT License - see the [LICENSE]() file for details.


Feel free to reach out with any questions or feedback. Happy job hunting!
