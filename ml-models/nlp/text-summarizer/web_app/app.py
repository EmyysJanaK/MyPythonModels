"""
Flask Web Application for Text Summarizer
Provides a user-friendly web interface for text summarization.
"""

from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
from flask_cors import CORS
import sys
import os
import json
import time
from datetime import datetime

# Add the src directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from summarizer.extractive_summarizer import ExtractiveSummarizer
    from summarizer.abstractive_summarizer import AbstractiveSummarizer
    from summarizer.hybrid_summarizer import HybridSummarizer
except ImportError as e:
    print(f"Warning: Could not import summarizers: {e}")
    print("Some features may not be available.")

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'  # Change this in production
CORS(app)

# Global variables for summarizers
summarizers = {}

def initialize_summarizers():
    """Initialize summarizers with error handling."""
    global summarizers
    
    try:
        summarizers['extractive'] = ExtractiveSummarizer()
        print("Extractive summarizer initialized")
    except Exception as e:
        print(f"Failed to initialize extractive summarizer: {e}")
    
    try:
        summarizers['abstractive'] = AbstractiveSummarizer()
        print("Abstractive summarizer initialized")
    except Exception as e:
        print(f"Failed to initialize abstractive summarizer: {e}")
    
    try:
        summarizers['hybrid'] = HybridSummarizer()
        print("Hybrid summarizer initialized")
    except Exception as e:
        print(f"Failed to initialize hybrid summarizer: {e}")

@app.route('/')
def index():
    """Main page with summarization interface."""
    return render_template('index.html')

@app.route('/api/summarize', methods=['POST'])
def api_summarize():
    """API endpoint for text summarization."""
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
        
        text = data['text'].strip()
        if not text:
            return jsonify({'error': 'Empty text provided'}), 400
        
        # Get parameters
        method = data.get('method', 'hybrid')
        summary_ratio = float(data.get('summary_ratio', 0.3))
        max_length = int(data.get('max_length', 150))
        min_length = int(data.get('min_length', 30))
        
        start_time = time.time()
        
        # Perform summarization
        if method == 'extractive':
            if 'extractive' not in summarizers:
                return jsonify({'error': 'Extractive summarizer not available'}), 500
            
            algorithm = data.get('extractive_algorithm', 'hybrid')
            summary = summarizers['extractive'].summarize(
                text,
                summary_ratio=summary_ratio,
                algorithm=algorithm
            )
            
            result = {
                'summary': summary,
                'method': 'extractive',
                'algorithm': algorithm,
                'processing_time': time.time() - start_time
            }
            
        elif method == 'abstractive':
            if 'abstractive' not in summarizers:
                return jsonify({'error': 'Abstractive summarizer not available'}), 500
            
            style = data.get('abstractive_style', 'balanced')
            summary = summarizers['abstractive'].summarize(
                text,
                max_length=max_length,
                min_length=min_length,
                summary_style=style
            )
            
            result = {
                'summary': summary,
                'method': 'abstractive',
                'style': style,
                'processing_time': time.time() - start_time
            }
            
        elif method == 'hybrid':
            if 'hybrid' not in summarizers:
                return jsonify({'error': 'Hybrid summarizer not available'}), 500
            
            approach = data.get('approach', 'auto')
            hybrid_result = summarizers['hybrid'].summarize(
                text,
                approach=approach,
                summary_ratio=summary_ratio,
                max_length=max_length,
                min_length=min_length
            )
            
            result = {
                'summary': hybrid_result['final_summary'],
                'method': 'hybrid',
                'approach_used': hybrid_result['method_used'],
                'confidence': hybrid_result['confidence'],
                'reasoning': hybrid_result['reasoning'],
                'all_summaries': hybrid_result['summaries'],
                'text_analysis': hybrid_result['analysis'],
                'processing_time': time.time() - start_time
            }
            
        else:
            return jsonify({'error': f'Unknown method: {method}'}), 400
        
        # Add statistics
        if 'hybrid' in summarizers:
            stats = summarizers['hybrid'].get_summary_statistics(text, result['summary'])
            result['statistics'] = stats
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/compare', methods=['POST'])
def api_compare():
    """API endpoint to compare different summarization methods."""
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
        
        text = data['text'].strip()
        if not text:
            return jsonify({'error': 'Empty text provided'}), 400
        
        start_time = time.time()
        results = {}
        
        # Extractive summary
        if 'extractive' in summarizers:
            try:
                summary = summarizers['extractive'].summarize(text)
                results['extractive'] = {
                    'summary': summary,
                    'success': True
                }
            except Exception as e:
                results['extractive'] = {
                    'error': str(e),
                    'success': False
                }
        
        # Abstractive summary
        if 'abstractive' in summarizers:
            try:
                summary = summarizers['abstractive'].summarize(text)
                results['abstractive'] = {
                    'summary': summary,
                    'success': True
                }
            except Exception as e:
                results['abstractive'] = {
                    'error': str(e),
                    'success': False
                }
        
        # Hybrid summary
        if 'hybrid' in summarizers:
            try:
                hybrid_result = summarizers['hybrid'].summarize(text)
                results['hybrid'] = {
                    'summary': hybrid_result['final_summary'],
                    'method_used': hybrid_result['method_used'],
                    'confidence': hybrid_result['confidence'],
                    'success': True
                }
            except Exception as e:
                results['hybrid'] = {
                    'error': str(e),
                    'success': False
                }
        
        return jsonify({
            'results': results,
            'processing_time': time.time() - start_time
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/analyze', methods=['POST'])
def api_analyze():
    """API endpoint for text analysis."""
    try:
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400
        
        text = data['text'].strip()
        if not text:
            return jsonify({'error': 'Empty text provided'}), 400
        
        if 'hybrid' not in summarizers:
            return jsonify({'error': 'Text analysis not available'}), 500
        
        analysis = summarizers['hybrid'].analyze_text_characteristics(text)
        recommendation = summarizers['hybrid'].recommend_approach(analysis)
        
        return jsonify({
            'analysis': analysis,
            'recommendation': {
                'approach': recommendation[0],
                'confidence': recommendation[1],
                'reasoning': recommendation[2]
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/models', methods=['GET'])
def api_models():
    """API endpoint to get available models."""
    models = {}
    
    if 'abstractive' in summarizers:
        try:
            models['abstractive'] = AbstractiveSummarizer.list_available_models()
        except:
            models['abstractive'] = {}
    
    return jsonify(models)

@app.route('/health')
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'available_summarizers': list(summarizers.keys())
    })

@app.errorhandler(404)
def not_found(error):
    return render_template('404.html'), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('500.html'), 500

if __name__ == '__main__':
    print("Initializing Text Summarizer Web App...")
    initialize_summarizers()
    
    if not summarizers:
        print("Warning: No summarizers available. Some features will not work.")
    
    print("Starting Flask application...")
    app.run(debug=True, host='0.0.0.0', port=5000)