#!/usr/bin/env python3
"""
Buy or Wait? - AI-Powered Financial Decision Agent
HackerRank Orchestrate September 2026
"""

import os
import sys
import csv
import re
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import anthropic

# Initialize Claude client
client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

# Token tracking
total_input_tokens = 0
total_output_tokens = 0
request_count = 0

def load_csv(filepath):
    """Load CSV file into list of dictionaries."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return list(csv.DictReader(f))

def parse_date(date_str):
    """Parse YYYY-MM-DD date string to datetime."""
    if not date_str or date_str.strip() == '':
        return None
    return datetime.strptime(date_str.strip(), '%Y-%m-%d')

def format_date(dt):
    """Format datetime to YYYY-MM-DD string."""
    return dt.strftime('%Y-%m-%d')

def to_decimal(value):
    """Convert string to Decimal, handling empty strings."""
    if value == '' or value is None:
        return None
    return Decimal(str(value))

def extract_amount_from_image(image_path):
    """Extract amount from image using Claude Vision."""
    global total_input_tokens, total_output_tokens
    
    try:
        import base64
        with open(image_path, 'rb') as f:
            image_data = base64.standard_b64encode(f.read()).decode('utf-8')
        
        message = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=500,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": image_data
                        }
                    },
                    {
                        "type": "text",
                        "text": "Extract the exact numerical amount from this financial document. Return ONLY the number without currency symbols or text. For example, if the document shows 'Amount: $1,234.56', return '1234.56'. If multiple amounts are shown, return the primary transaction amount."
                    }
                ]
            }]
        )
        
        total_input_tokens += message.usage.input_tokens
        total_output_tokens += message.usage.output_tokens
        
        amount_str = message.content[0].text.strip()
        # Clean up the response
        amount_str = re.sub(r'[^\d.]', '', amount_str)
        return Decimal(amount_str) if amount_str else None
    except Exception as e:
        print(f"Error extracting amount from {image_path}: {e}", file=sys.stderr)
        return None

def build_financial_context(user_id, request_id, request_date, profiles, events, messages, images, exchange_rates, payment_options):
    """Build comprehensive financial context for a user and request."""
    
    # Get user profile
    profile = next((p for p in profiles if p['user_id'] == user_id), None)
    if not profile:
        return None
    
    # Get user events
    user_events = [e for e in events if e['user_id'] == user_id]
    
    # Get messages for user and request
    user_messages = [m for m in messages if m['user_id'] == user_id]
    request_messages = [m for m in user_messages if m['request_id'] == request_id]
    
    # Get images for user and request
    user_images = [img for img in images if img['user_id'] == user_id]
    request_images = [img for img in user_images if img['request_id'] == request_id]
    
    # Get payment options for request
    request_payment_options = [po for po in payment_options if po['request_id'] == request_id]
    
    # Process events and extract amounts from images where needed
    processed_events = []
    for event in user_events:
        event_copy = event.copy()
        if event['amount'] == '' or event['amount'] is None:
            # Find related image
            related_images = [img for img in images if img['related_event_id'] == event['event_id']]
            if related_images:
                image_id = related_images[0]['image_id']
                image_path = f"dataset/media/images/{image_id}.png"
                if os.path.exists(image_path):
                    amount = extract_amount_from_image(image_path)
                    if amount:
                        event_copy['amount'] = str(amount)
        processed_events.append(event_copy)
    
    return {
        'profile': profile,
        'events': processed_events,
        'messages': user_messages,
        'request_messages': request_messages,
        'images': user_images,
        'request_images': request_images,
        'payment_options': request_payment_options
    }

def call_claude_for_decision(request, context):
    """Use Claude to make financial decision based on all context."""
    global total_input_tokens, total_output_tokens, request_count
    
    # Build comprehensive prompt
    prompt = f"""You are an expert financial advisor analyzing a user's ability to afford an expense.

USER REQUEST:
Request ID: {request['request_id']}
User ID: {request['user_id']}
Request Date: {request['request_date']}
Request Type: {request['request_type']}
Requested Amount: {request['requested_amount']} (user's home currency)
Desired Completion Date: {request['desired_completion_date']}
Allows Partial Payment: {request['allows_partial_payment']}
Request Text: {request['request_text']}

USER PROFILE:
Home Currency: {context['profile']['home_currency']}
Current Available Balance: {context['profile']['current_available_balance']}
Minimum Balance to Keep: {context['profile']['minimum_balance_to_keep']}
Financial Priorities: {context['profile']['financial_priorities']}
Protected Expense Categories: {context['profile']['expense_categories_to_protect']}
Categories User Will Reduce: {context['profile']['expense_categories_user_is_willing_to_reduce']}
Categories User Will Stop: {context['profile']['expense_categories_user_is_willing_to_stop']}
Payment Methods User Will Consider: {context['profile']['payment_methods_user_will_consider']}
Max Installment Months: {context['profile']['max_installment_months']}

FINANCIAL EVENTS (recent 6 months + future):
{chr(10).join([f"- {e['event_id']}: {e['event_type']} | {e['description']} | {e['category']} | {e['direction']} | Amount: {e['amount']} {e['currency']} | Date: {e['event_date']} | Settlement: {e['settlement_date']} | Status: {e['status']} | Flexibility: {e['flexibility']}" for e in context['events'][-100:]])}

RELEVANT MESSAGES:
{chr(10).join([f"- From {m['source_type']} at {m['sent_at']}: {m['message_text']}" for m in context['request_messages']]) if context['request_messages'] else 'No relevant messages'}

PAYMENT OPTIONS AVAILABLE:
{chr(10).join([f"- Option {po['payment_option_id']}: {po['payment_method']} | {po['number_of_payments']} payments of {po['payment_amount']} starting {po['first_payment_date']} | Total: {po['total_payable_amount']}" for po in context['payment_options']])}

TASK:
Analyze this financial situation comprehensively and provide a recommendation. Consider:
1. Recurring income and expenses (identify patterns from historical events)
2. Pending transactions (only settled and confirmed income counts)
3. Essential vs flexible spending
4. 90-day cash flow forecast
5. Minimum balance protection throughout forecast period
6. User's payment preferences and priorities
7. Available payment options
8. Messages that might confirm, amend, cancel, or clarify events

OUTPUT FORMAT (JSON):
{{
    "amount_safe_to_pay": <number between 0 and requested_amount>,
    "affordability_status": "<affordable_now|affordable_with_plan|affordable_later|not_affordable>",
    "recommended_payment_method": "<full_payment|partial_payment|installments|wait|not_recommended>",
    "payment_plan": "<YYYY-MM-DD:amount|YYYY-MM-DD:amount or 'none'>",
    "earliest_date_for_full_payment": "<YYYY-MM-DD or empty>",
    "spending_changes_needed": "<stop:event_id|reduce_to:event_id:amount or 'none'>",
    "decision_explanation": "<concise explanation>"
}}

CRITICAL RULES:
- amount_safe_to_pay must be <= requested_amount
- For affordable_now, earliest_date_for_full_payment must equal request_date
- For partial_payment: use exactly 2 payments (amount_safe_to_pay on request_date, remainder on earliest_date_for_full_payment), only if allows_partial_payment=true
- For installments: must match exactly one of the payment options provided
- Only suggest stopping/reducing flexible expenses that user is willing to change
- Balance must never drop below minimum_balance_to_keep at any point in 90-day forecast
- Prefer solutions that: complete by deadline, avoid spending changes, minimize total cost, start earlier, use fewer payments

Return ONLY valid JSON, no other text."""
    
    try:
        message = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=2000,
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )
        
        total_input_tokens += message.usage.input_tokens
        total_output_tokens += message.usage.output_tokens
        request_count += 1
        
        response_text = message.content[0].text.strip()
        
        # Extract JSON from response
        import json
        # Try to find JSON in response
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group(0))
            return result
        else:
            print(f"Warning: Could not parse JSON from response for {request['request_id']}", file=sys.stderr)
            return None
            
    except Exception as e:
        print(f"Error calling Claude for {request['request_id']}: {e}", file=sys.stderr)
        return None

def process_request(request, profiles, events, messages, images, exchange_rates, payment_options):
    """Process a single request and return prediction."""
    
    # Build context
    context = build_financial_context(
        request['user_id'],
        request['request_id'],
        request['request_date'],
        profiles, events, messages, images, exchange_rates, payment_options
    )
    
    if not context:
        # Fallback if context building fails
        return {
            'request_id': request['request_id'],
            'amount_safe_to_pay': '0',
            'affordability_status': 'not_affordable',
            'recommended_payment_method': 'not_recommended',
            'payment_plan': 'none',
            'earliest_date_for_full_payment': '',
            'spending_changes_needed': 'none',
            'decision_explanation': 'Unable to build financial context for analysis.'
        }
    
    # Get AI decision
    decision = call_claude_for_decision(request, context)
    
    if not decision:
        # Fallback if AI call fails
        return {
            'request_id': request['request_id'],
            'amount_safe_to_pay': '0',
            'affordability_status': 'not_affordable',
            'recommended_payment_method': 'not_recommended',
            'payment_plan': 'none',
            'earliest_date_for_full_payment': '',
            'spending_changes_needed': 'none',
            'decision_explanation': 'AI analysis failed. Please review manually.'
        }
    
    # Format and return
    return {
        'request_id': request['request_id'],
        'amount_safe_to_pay': str(decision.get('amount_safe_to_pay', '0')),
        'affordability_status': decision.get('affordability_status', 'not_affordable'),
        'recommended_payment_method': decision.get('recommended_payment_method', 'not_recommended'),
        'payment_plan': decision.get('payment_plan', 'none'),
        'earliest_date_for_full_payment': decision.get('earliest_date_for_full_payment', ''),
        'spending_changes_needed': decision.get('spending_changes_needed', 'none'),
        'decision_explanation': decision.get('decision_explanation', '')
    }

def main():
    """Main entry point."""
    global total_input_tokens, total_output_tokens, request_count
    
    print("Loading dataset...")
    
    # Load all data
    profiles = load_csv('dataset/financial_profiles.csv')
    events = load_csv('dataset/financial_events.csv')
    messages = load_csv('dataset/messages.csv')
    images = load_csv('dataset/images.csv')
    exchange_rates = load_csv('dataset/exchange_rates.csv')
    payment_options = load_csv('dataset/request_payment_options.csv')
    requests = load_csv('dataset/requests.csv')
    
    print(f"Loaded {len(requests)} requests to process")
    
    # Process each request
    results = []
    for i, request in enumerate(requests, 1):
        print(f"Processing request {i}/{len(requests)}: {request['request_id']}")
        result = process_request(request, profiles, events, messages, images, exchange_rates, payment_options)
        results.append(result)
    
    # Write output
    output_file = 'output.csv'
    fieldnames = [
        'request_id',
        'amount_safe_to_pay',
        'affordability_status',
        'recommended_payment_method',
        'payment_plan',
        'earliest_date_for_full_payment',
        'spending_changes_needed',
        'decision_explanation'
    ]
    
    with open(output_file, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\nOutput written to {output_file}")
    print(f"\nToken Usage Summary:")
    print(f"Total Requests: {request_count}")
    print(f"Total Input Tokens: {total_input_tokens}")
    print(f"Total Output Tokens: {total_output_tokens}")
    print(f"Average Input Tokens per Request: {total_input_tokens / request_count if request_count > 0 else 0:.2f}")
    print(f"Average Output Tokens per Request: {total_output_tokens / request_count if request_count > 0 else 0:.2f}")
    
    # Write usage report
    os.makedirs('code/evaluation', exist_ok=True)
    with open('code/evaluation/usage_report.md', 'w') as f:
        f.write("# Token Usage and Cost Report\n\n")
        f.write(f"## Model Information\n")
        f.write(f"- Provider: Anthropic\n")
        f.write(f"- Model: claude-3-5-sonnet-20241022\n\n")
        f.write(f"## Usage Statistics\n")
        f.write(f"- Total Requests Processed: {len(requests)}\n")
        f.write(f"- Total Model Calls: {request_count}\n")
        f.write(f"- Total Input Tokens: {total_input_tokens:,}\n")
        f.write(f"- Total Output Tokens: {total_output_tokens:,}\n")
        f.write(f"- Total Tokens: {total_input_tokens + total_output_tokens:,}\n")
        f.write(f"- Average Input Tokens per Request: {total_input_tokens / len(requests) if len(requests) > 0 else 0:.2f}\n")
        f.write(f"- Average Output Tokens per Request: {total_output_tokens / len(requests) if len(requests) > 0 else 0:.2f}\n\n")
        f.write(f"## Cost Estimation\n")
        # Claude 3.5 Sonnet pricing (as of 2024)
        input_cost_per_mtok = 3.00  # $3 per million input tokens
        output_cost_per_mtok = 15.00  # $15 per million output tokens
        total_input_cost = (total_input_tokens / 1_000_000) * input_cost_per_mtok
        total_output_cost = (total_output_tokens / 1_000_000) * output_cost_per_mtok
        total_cost = total_input_cost + total_output_cost
        
        f.write(f"- Input Cost: ${total_input_cost:.2f}\n")
        f.write(f"- Output Cost: ${total_output_cost:.2f}\n")
        f.write(f"- Total Estimated Cost: ${total_cost:.2f}\n")
        f.write(f"- Cost per Request: ${total_cost / len(requests) if len(requests) > 0 else 0:.4f}\n")
    
    print(f"\nUsage report written to code/evaluation/usage_report.md")

if __name__ == '__main__':
    main()
