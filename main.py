import yaml
import pandas as pd
from argparse import Namespace
from tqdm import tqdm
import warnings
from utils import load_model, prepare_dataset, generate, detect

# Ignore all future warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def load_config(file_path):
    """
    Load configuration settings from a YAML file.

    Args:
        file_path (str): The path to the YAML configuration file.

    Returns:
        Namespace: A namespace containing the configuration settings.
    """
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)


def process_document(input_text, id, args, model, tokenizer, device):
    """
    Process a document and generate results.

    Args:
        input_text (str): The input text to process.
        id (int): The ID of the document.
        args (Namespace): Command-line arguments.
        model: The model for text generation.
        tokenizer: The tokenizer for text processing.
        device: The device for computation.

    Returns:
        dict: A dictionary containing the processing results.
    """
    args.default_prompt = input_text
    _, _, decoded_output_without_watermark, decoded_output_with_watermark, num_token_without_watermark, num_token_with_watermark, _ = generate(
        input_text,
        args,
        model=model,
        device=device,
        tokenizer=tokenizer)

    without_watermark_detection_result = detect(decoded_output_without_watermark,
                                                args,
                                                device=device,
                                                tokenizer=tokenizer)[0]
    with_watermark_detection_result = detect(decoded_output_with_watermark,
                                             args,
                                             device=device,
                                             tokenizer=tokenizer)[0]

    result = {
        'id': str(id),
        'WO_num_tokens_scored': without_watermark_detection_result['num_tokens_scored'],
        'WO_num_green_tokens': without_watermark_detection_result['num_green_tokens'],
        'WO_green_fraction': without_watermark_detection_result['green_fraction'],
        'WO_z_score': without_watermark_detection_result['z_score'],
        'WO_p_value': without_watermark_detection_result['p_value'],
        'WO_prediction': without_watermark_detection_result['prediction'],
        'WO_num_output_token': num_token_without_watermark,
        'W_num_tokens_scored': with_watermark_detection_result['num_tokens_scored'],
        'W_num_green_tokens': with_watermark_detection_result['num_green_tokens'],
        'W_green_fraction': with_watermark_detection_result['green_fraction'],
        'W_z_score': with_watermark_detection_result['z_score'],
        'W_p_value': with_watermark_detection_result['p_value'],
        'W_prediction': with_watermark_detection_result['prediction'],
        'W_num_output_token': num_token_with_watermark,
    }

    return result


def main(args):
    # Parse the 'normalizers' argument into a list
    args.normalizers = args.normalizers.split(",") if args.normalizers else []
    print(args)

    # Load the model, tokenizer, and device
    model, tokenizer, device = load_model(args)

    # Prepare the dataset
    dataset = prepare_dataset(args)

    # Read existing results from 'result.csv' if it exists, or create an empty DataFrame
    try:
        df_existing = pd.read_csv('result.csv')
    except FileNotFoundError:
        df_existing = pd.DataFrame(columns=['id'])

    df = pd.DataFrame()

    for input_text, id in tqdm(zip(dataset['document'], dataset['id'])):
        # Check if the ID already exists in the existing DataFrame
        if str(id) in df_existing['id'].values:
            continue  # Skip processing this ID

        result = process_document(input_text, id, args, model, tokenizer, device)

        # Append the result dictionary to a DataFrame
        df_result = pd.DataFrame([result])

        # Append the result DataFrame to the main DataFrame
        df = pd.concat([df, df_result], ignore_index=True)

    # Combine the existing DataFrame and the new results DataFrame
    df = pd.concat([df_existing, df], ignore_index=True)

    # Save the updated DataFrame as a CSV file
    df.to_csv(args.csv_path, index=False)


if __name__ == "__main__":
    config_path = 'config.yaml'
    # Load YAML file to initialize the 'args' Namespace
    config = load_config(config_path)
    args = Namespace(**config)
    main(args)
