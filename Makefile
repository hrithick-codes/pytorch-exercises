clean:
	echo "Cleaning .pyc files and __pycache__ directories..."
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -delete
	echo "Cleanup done."

format:
	echo "Sorting imports.."
	isort .
	echo "Sorting complete."

	echo "Formatting Python files with black..."
	black .
	echo "Formatting complete."

	echo "Running flake8 for linting..."
	flake8 .
	echo "Linting complete."

train-rnn-sentiment-classifier:
	@echo "==============================================="
	@echo "🚀 Starting RNN Sentiment Classifier Training 🚀"
	@echo "This may take a while, so grab a coffee! ☕"
	@echo "Logs and output will be displayed below 👇"
	@echo "==============================================="
	PYTHONPATH=. python3 training_scripts/rnn_classifier.py

train-rnn-lm:
	PYTHONPATH=. python3 training_scripts/rnn_lm.py
