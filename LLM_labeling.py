import pandas as pd
import asyncio
import json
import time
from openai import AsyncOpenAI

# use key from openai.key
with open('openai.key', 'r') as key_file:
    api_key = key_file.read().strip()

# Create an async OpenAI client
client = AsyncOpenAI(api_key=api_key)

# Create a semaphore to limit concurrent requests
# Adjust this value based on your needs - lower for stricter rate limiting
MAX_CONCURRENT_REQUESTS = 100
semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

# Retry decorator for async functions
async def async_retry_with_backoff(func, *args, **kwargs):
    max_retries = 6
    min_wait = 1
    max_wait = 60
    
    for attempt in range(max_retries):
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            # Check if this is the last attempt
            if attempt == max_retries - 1:
                raise e
            
            # Calculate wait time with exponential backoff and jitter
            wait_time = min(max_wait, min_wait * (2 ** attempt))
            wait_time = wait_time * (0.5 + 0.5 * (time.time() % 1))  # Add jitter
            
            print(f"Rate limit hit or error occurred. Retrying in {wait_time:.2f} seconds... (Attempt {attempt+1}/{max_retries})")
            await asyncio.sleep(wait_time)

async def classify_text(text, topic):
    # Use the semaphore to limit concurrent requests
    async with semaphore:
        # Use the retry mechanism
        return await async_retry_with_backoff(_classify_text, text, topic)

async def _classify_text(text, topic):
    response = await client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {
                "role": "system",
                "content": [
                    {
                        "type": "input_text",
                        "text": "You are an expert text analysis model. Your task is to determine if a given paragraph is opinionated or un-opinionated based on the definitions provided below. You will also provide a brief analysis explaining your reasoning.\n\n**Definitions:**\n\n* **Opinionated:** Text that expresses a evaluative judgment, or stance about a topic. This includes subjective assessments and viewpoints that can be debated or disagreed with.\n* **Un-opinionated:** Text that presents description (factual information, objective descriptions, or widely accepted knowledge), experience  (e.g. hot/cold) unrelated to evaluation (opinionated: good/bad), inquiery with no apriori evaluative assumption, etc. This type of text is typically verifiable and does not express personal feelings or judgments. \n\n**Crucially, for this task, descriptions of personal experiences are NOT considered opinionated text, even if they inform a personal judgment elsewhere.** A description of an experience reports on what happened or how something felt *to the individual* without necessarily making a general judgment or evaluation about the subject itself.\n\n**Examples:**\n\n* **Opinionated:** \"The new public transportation policy is a disaster for commuters.\" (Evaluates the policy negatively)\n* **Opinionated:** \"Restaurant A is the best dining experience in the city.\" (Subjective judgment about a restaurant)\n* **Un-opinionated:** \"The policy X is believed to have led to an increase in traffic congestion.\" (Reports on a belief/claim/hopothesis, not stating it as a personal opinion)\n* **Un-opinionated:** \"The sky is blue.\" (Factual statement)\n* **Un-opinionated (Description of Experience):** \"For me, the car felt faster than the bus during my commute this morning.\" (Reports a personal sensation/experience without a general judgment on the car or bus's speed such as \"faster is better\".)\n* **Opinionated (Judgment based on Experience):** \"Because the car felt faster than the bus this morning, I believe cars are a superior mode of transport.\" (Connects an experience to a general, debatable judgment)\n\n**Instructions:**\n\nAnalyze the provided paragraph carefully. Determine if the primary nature of the text is opinionated or un-opinionated based on the definitions above, paying close attention to the distinction regarding personal experiences.\n\nProvide your output in two parts: first, a brief analysis explaining your reasoning for the classification, and second, the final classification category."
                    }
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": f"Topic Group: {topic}\n\n Text: {text}"
                    }
                ]
            }
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "text_analysis_classification",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "analysis": {
                            "type": "string",
                            "description": "A brief analysis explaining the reasoning for classifying the text as opinionated or un-opinionated."
                        },
                        "classification": {
                            "type": "string",
                            "enum": [
                                "opinionated",
                                "un-opinionated"
                            ],
                            "description": "The final classification of the paragraph."
                        }
                    },
                    "required": [
                        "analysis",
                        "classification"
                    ],
                    "additionalProperties": False
                }
            }
        },
        reasoning={},
        tools=[],
        temperature=1,
        max_output_tokens=2048,
        top_p=1,
        store=True
    )
    
    # Extract the JSON content from the response
    json_text = response.output[0].content[0].text
    result = json.loads(json_text)
    return result

# Create a new dataframe to store results
async def main():
    # Read from 20_newsgroup_split.csv
    df = pd.read_csv('raw_20newsgroup.csv')
    df = df.dropna(subset=['text'])
    df = df[df['text'].str.strip() != '']
    df = df[df['text'].str.len() > 4]
    
    # Create a new dataframe with only the columns we need
    results_df = pd.DataFrame(columns=['classification', 'topic', 'text', 'analysis'])
    
    # # For testing purpose, Randomly select N rows from the dataframe
    # n_rows = 20
    # rows = df.sample(n=n_rows, random_state=1)
    
    rows = df.head(10000)
    n_rows = len(rows)
    
    tasks = []
    
    n = 1
    # Create tasks for all API calls without waiting for responses
    for index, row in rows.iterrows():
        print(f"Starting task for row {index+1}, {n}/{n_rows}...")
        n += 1
        text = row['text']
        topic = row['title']
        
        # Create a task for each API call
        task = asyncio.create_task(process_row(index, text, topic))
        tasks.append((index, n, task))
    
    # Wait for all tasks to complete
    print("Waiting for all API calls to complete...")
    results = []
    for index, n, task in tasks:
        try:
            result = await task
            results.append(result)
            print(f"Completed row {index}, {n}/{n_rows}: {result['classification']}")
        except Exception as e:
            print(f"Error processing row {index}: {e}")
    
    # Add all results to the dataframe
    for result in results:
        results_df = pd.concat([results_df, pd.DataFrame({
            'classification': [result['classification']],
            'topic': [result['topic']],
            'text': [result['text']],
            'analysis': [result['analysis']],
        })], ignore_index=True)
    
    # reorient the columns to be topic,text,classification,analysis
    results_df = results_df[['topic', 'text', 'classification', 'analysis']]
    
    # Save results to CSV
    results_df.to_csv('classification_results.csv', index=False)
    print(f"\nResults saved to classification_results.csv")
    print("\nSample of results:")
    print(results_df[['topic', 'classification', 'analysis']].head())

async def process_row(index, text, topic):
    try:
        # Get classification from OpenAI
        result = await classify_text(text, topic)
        return {
            'classification': result['classification'],
            'topic': topic,
            'text': text,
            'analysis': result['analysis'],
        }
    except Exception as e:
        print(f"Error in process_row {index}: {e}")
        raise e

if __name__ == "__main__":
    asyncio.run(main())
