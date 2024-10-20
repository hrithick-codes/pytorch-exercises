clean:
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -delete

format:
	black .
	isort .

train-rnn-sentiment-classifier:
	@echo "==============================================="
	@echo "🚀 Starting RNN Sentiment Classifier Training 🚀"
	@echo "This may take a while, so grab a coffee! ☕"
	@echo "Logs and output will be displayed below 👇"
	@echo "==============================================="
	PYTHONPATH=. python3 training_scripts/rnn_classifier.py

train-rnn-lm:
	PYTHONPATH=. python3 training_scripts/rnn_lm.py
