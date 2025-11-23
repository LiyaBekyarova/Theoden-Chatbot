# Create the venv
python3 -m venv venv

# Activate it (zsh)
source venv/bin/activate

# Now inside the venv, "python" will work automatically
python --version    # ← this will now work
pip install torch nltk 