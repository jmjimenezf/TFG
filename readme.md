Scripts for my TFG experiments.

Evaluation of LLMS
- tinyllama:1.1b-chat-v1-q4_K_M
- gemma2:9b-instruct-q4_K_M
- mistral:7b-instruct-q4_K_M
- llama3.1:8b-instruct-q4_K_M

All models run locally in Ollama with 0 temperature and quantizized with 4_K_M.
 
Sentiment analysis and emotion detection with corpus:
- SST2  (Socher et al., 2013)
- GoodNewsEveryone (Bostan et al., 2020)

# Information
## Requirements
- You can find the requirements for python in requirements.txt
- Ollama must be installed and accesible from localhost
- Huggingface API key

## Settings
In file config_SA.yaml you can configure the models and temperature to run, you need to install them manually in Ollama first. You can also select the corpus and establish the prompt and context.
## Corpus
First corpus SST2 is accessed via Huggingface API, you need to export your key into an env variable called: HF_TOKEN. The GNE corpus is accessed locally in corpus folder.

# Sources

Bostan, Laura Ana Maria, et al. «GoodNewsEveryone: A Corpus of News Headlines Annotated with Emotions, Semantic Roles, and Reader Perception». Proceedings of the Twelfth Language Resources and Evaluation Conference, editado por Nicoletta Calzolari et al., European Language Resources Association, 2020, pp. 1554-66. ACLWeb, https://aclanthology.org/2020.lrec-1.194/.

Socher, Richard, et al. «Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank». Proceedings of the 2013 Conference on Empirical Methods in Natural Language Processing, editado por David Yarowsky et al., Association for Computational Linguistics, 2013, pp. 1631-42. ACLWeb, https://aclanthology.org/D13-1170/.
