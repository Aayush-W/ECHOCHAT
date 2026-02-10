import re
from datetime import datetime
from typing import List, Dict


def _is_emoji_char(ch: str) -> bool:
    cp = ord(ch)
    return (
        0x1F1E6 <= cp <= 0x1F1FF
        or 0x1F300 <= cp <= 0x1FAFF
        or 0x2600 <= cp <= 0x27BF
    )

def parse_whatsapp_chat(file_path: str) -> List[Dict]:
    """
    Parse WhatsApp exported chat (.txt) into structured data.
    
    Handles:
    - Unicode, emojis, Hinglish, Marathi-English
    - Empty messages
    - System messages (filtered)
    - Timestamps
    - Multi-line messages
    - Media descriptions
    
    Returns:
    List of dicts with: timestamp, sender, message, length, has_emoji
    """
    
    # WhatsApp timestamp pattern: DD/MM/YYYY, HH:MM am/pm - Sender: Message
    # Handles non-breaking space (\u202f) used by WhatsApp
    timestamp_pattern = r'^(\d{1,2}/\d{1,2}/\d{4}),\s(\d{1,2}:\d{2}\s(?:am|pm))\s*[-\u2013]\s*(.+?):\s(.*)$'
    
    system_keywords = {
        'Messages you send to this group are now encrypted',
        'security code changed',
        'media omitted',
        'this messages was deleted',
        'you created this group',
        'added',
        'removed',
        'changed this group\'s icon',
        'changed the subject',
        'left',
    }
    
    messages = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return []
    except UnicodeDecodeError:
        print(f"Error: File encoding issue. Try UTF-8 with errors='ignore'")
        return []
    
    for idx, line in enumerate(lines):
        line = line.rstrip('\n')
        
        # Check if line is a timestamp + sender
        match = re.match(timestamp_pattern, line, re.IGNORECASE)
        
        if match:
            timestamp_str = f"{match.group(1)} {match.group(2)}"
            sender = match.group(3).strip()
            message = match.group(4).strip()
            
            # Parse timestamp
            try:
                timestamp = datetime.strptime(timestamp_str, "%d/%m/%Y %I:%M %p")
            except ValueError:
                print(f"Warning: Could not parse timestamp '{timestamp_str}'")
                continue
            
            # Check if system message
            if any(keyword.lower() in message.lower() for keyword in system_keywords):
                continue
            
            # Skip empty messages
            if not message:
                continue
            
            # Count emojis (emoji codepoint check)
            has_emoji = any(_is_emoji_char(ch) for ch in message)
            
            messages.append({
                'timestamp': timestamp,
                'sender': sender,
                'message': message,
                'length': len(message),
                'has_emoji': has_emoji,
            })
        
        else:
            # Continuation of previous message (multi-line)
            if not line.strip():
                continue
            if messages:
                messages[-1]['message'] += '\n' + line
                messages[-1]['length'] = len(messages[-1]['message'])
            else:
                # Orphaned line at start of file
                print(f"Warning: Orphaned line (no previous message): {line[:50]}")
    
    return messages
