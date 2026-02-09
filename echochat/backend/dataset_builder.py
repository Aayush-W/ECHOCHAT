from typing import List, Dict, Tuple
import json

ECHOPERSON_NAME = "Lady Parus"
USER_NAME = "Aayush W⚡"


def build_datasets(
    messages: List[Dict],
    echo_person: str = "Lady Parus"
) -> Tuple[List[Dict], List[Dict]]:
    """
    Build training and memory datasets from parsed messages.
    
    TRAINING DATA:
    - Pairs: (User Message) -> (Echo Person Response)
    - Used for fine-tuning LLM
    - Only includes messages from echo_person addressing others
    
    MEMORY DATA:
    - All messages from echo_person with timestamps
    - Used for semantic retrieval
    
    Args:
        messages: List from parse_whatsapp_chat()
        echo_person: Name of the person to simulate
    
    Returns:
        (training_data, memory_data)
    """
    
    training_data = []
    memory_data = []
    
    # Filter messages from echo_person only
    echo_messages = [msg for msg in messages if msg['sender'] == echo_person]
    
    if not echo_messages:
        print(f"Warning: No messages found from '{echo_person}'")
        return [], []
    
    # Build memory data (all echo_person messages)
    memory_data = [
        {
            'text': msg['message'],
            'timestamp': msg['timestamp'].isoformat(),
            'length': msg['length'],
            'has_emoji': msg['has_emoji'],
            'context': {
                'word_count': len(msg['message'].split()),
                'contains_code': any(char in msg['message'] for char in ['<', '>', '{', '}']),
                'is_question': msg['message'].strip().endswith('?'),
            }
        }
        for msg in echo_messages
    ]
    
    # Build training data (message pairs)
    for i in range(len(messages) - 1):
        current_msg = messages[i]
        next_msg = messages[i + 1]
        
        # Skip if next message is not from echo_person
        if next_msg['sender'] != echo_person:
            continue
        
        # Skip if current and next are from same sender (not a reply pattern)
        if current_msg['sender'] == echo_person:
            continue
        
        # Skip very short inputs or outputs
        if len(current_msg['message']) < 2 or len(next_msg['message']) < 1:
            continue
        
        training_pair = {
            'instruction': f'Reply as {echo_person} in his usual conversational style',
            'input': current_msg['message'],
            'output': next_msg['message'],
            'metadata': {
                'input_sender': current_msg['sender'],
                'timestamp_input': current_msg['timestamp'].isoformat(),
                'timestamp_output': next_msg['timestamp'].isoformat(),
                'input_length': current_msg['length'],
                'output_length': next_msg['length'],
                'input_has_emoji': current_msg['has_emoji'],
                'output_has_emoji': next_msg['has_emoji'],
            }
        }
        
        training_data.append(training_pair)
    
    print(f"Built {len(training_data)} training pairs")
    print(f"Built {len(memory_data)} memory entries")
    
    return training_data, memory_data


def save_datasets(
    training_data: List[Dict],
    memory_data: List[Dict],
    training_path: str = "data/training_data.jsonl",
    memory_path: str = "data/memory_data.json"
) -> None:
    """
    Save datasets to files.
    
    TRAINING: JSONL format (one JSON per line, for fine-tuning)
    MEMORY: JSON format (array, for semantic search)
    """
    
    # Save training data as JSONL
    try:
        with open(training_path, 'w', encoding='utf-8') as f:
            for pair in training_data:
                f.write(json.dumps(pair, ensure_ascii=False) + '\n')
        print(f"✅ Saved {len(training_data)} training pairs to {training_path}")
    except IOError as e:
        print(f"❌ Error saving training data: {e}")
    
    # Save memory data as JSON
    try:
        with open(memory_path, 'w', encoding='utf-8') as f:
            json.dump(memory_data, f, ensure_ascii=False, indent=2)
        print(f"✅ Saved {len(memory_data)} memory entries to {memory_path}")
    except IOError as e:
        print(f"❌ Error saving memory data: {e}")


def load_training_data(path: str = "data/training_data.jsonl") -> List[Dict]:
    """Load training data from JSONL file."""
    data = []
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
    except FileNotFoundError:
        print(f"Error: Training file not found at {path}")
    return data


def load_memory_data(path: str = "data/memory_data.json") -> List[Dict]:
    """Load memory data from JSON file."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Memory file not found at {path}")
        return []
