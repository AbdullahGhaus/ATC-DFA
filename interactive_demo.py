"""
Interactive Demo for DFA Spam Detector
Allows users to test messages in real-time
"""
from classifier import build_keyword_dfa, classify_message
import sys


def load_keywords(keywords_file=None):
    """
    Load keywords from file or use default
    """
    if keywords_file:
        try:
            with open(keywords_file, 'r') as f:
                keywords = [line.strip().lower() for line in f if line.strip()]
            return keywords
        except FileNotFoundError:
            print(f"Warning: Keywords file '{keywords_file}' not found. Using defaults.")
    
    return ["win", "free", "congratulations", "prize", "urgent", "click", "limited"]


def interactive_classify(keywords):
    """
    Interactive classification interface
    """
    dfas = [build_keyword_dfa(k) for k in keywords]
    
    print("=" * 70)
    print("DFA SPAM DETECTOR - INTERACTIVE DEMO")
    print("=" * 70)
    print(f"\nActive Keywords: {', '.join([kw.upper() for kw in keywords])}")
    print("\nEnter messages to classify (type 'quit' or 'exit' to stop)")
    print("-" * 70)
    
    while True:
        try:
            message = input("\nEnter message: ").strip()
            
            if message.lower() in ['quit', 'exit', 'q']:
                print("\nThank you for using DFA Spam Detector!")
                break
            
            if not message:
                print("Please enter a message.")
                continue
            
            # Classify
            result = classify_message(dfas, message)
            
            # Find which keyword(s) triggered
            message_lower = message.lower()
            triggered_keywords = [kw for kw in keywords if kw in message_lower]
            
            # Display result
            print(f"\n{'='*70}")
            print(f"Result: {result.upper()}")
            if triggered_keywords:
                print(f"Triggered Keywords: {', '.join([kw.upper() for kw in triggered_keywords])}")
            else:
                print("No spam keywords detected.")
            print(f"{'='*70}")
            
        except KeyboardInterrupt:
            print("\n\nExiting...")
            break
        except Exception as e:
            print(f"Error: {e}")


def batch_classify(keywords, messages):
    """
    Classify multiple messages at once
    """
    dfas = [build_keyword_dfa(k) for k in keywords]
    
    results = []
    for message in messages:
        result = classify_message(dfas, message)
        message_lower = message.lower()
        triggered = [kw for kw in keywords if kw in message_lower]
        results.append({
            'message': message,
            'prediction': result,
            'keywords': triggered
        })
    
    return results


if __name__ == "__main__":
    keywords = load_keywords()
    
    if len(sys.argv) > 1:
        # Batch mode
        messages = sys.argv[1:]
        results = batch_classify(keywords, messages)
        print("\nBatch Classification Results:")
        print("=" * 70)
        for r in results:
            print(f"\nMessage: {r['message']}")
            print(f"Prediction: {r['prediction'].upper()}")
            if r['keywords']:
                print(f"Keywords: {', '.join(r['keywords'])}")
    else:
        # Interactive mode
        interactive_classify(keywords)

