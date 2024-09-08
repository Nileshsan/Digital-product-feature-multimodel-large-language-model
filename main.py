

from flask import app, request, jsonify
import pylance

@app.route('/process-screenshots', methods=['POST'])
def process_screenshots():
  context = request.form.get('context', '')
  screenshots = request.files.getlist('screenshots')

  # Process screenshots and generate testing instructions using the LLM
  testing_instructions = generate_testing_instructions(screenshots, context)

  return jsonify({'testingInstructions': testing_instructions})