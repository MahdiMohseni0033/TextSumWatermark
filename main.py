from pprint import pprint

import yaml
from argparse import Namespace
from tqdm import tqdm
from utils import *
import pandas as pd

def main2(args):
    args.normalizers = (args.normalizers.split(",") if args.normalizers else [])
    print(args)

    model, tokenizer, device = load_model(args)
    dataset = prepare_dataset(dataset_name='xsum', limit=4)
    df = pd.DataFrame()
    for input_text, id in tqdm(zip(dataset['document'], dataset['id'])):
        args.default_prompt = input_text
        _, _, decoded_output_without_watermark, decoded_output_with_watermark, _ = generate(input_text,
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

        result = {}
        result['id'] = id
        result['WO_num_tokens_scored'] = without_watermark_detection_result['num_tokens_scored']
        result['WO_num_green_tokens'] = without_watermark_detection_result['num_green_tokens']
        result['WO_green_fraction'] = without_watermark_detection_result['green_fraction']
        result['WO_z_score'] = without_watermark_detection_result['z_score']
        result['WO_p_value'] = without_watermark_detection_result['p_value']
        result['WO_prediction'] = without_watermark_detection_result['prediction']

        result['W_num_tokens_scored'] = with_watermark_detection_result['num_tokens_scored']
        result['W_num_green_tokens']= with_watermark_detection_result['num_green_tokens']
        result['W_green_fraction']= with_watermark_detection_result['green_fraction']
        result['W_z_score']= with_watermark_detection_result['z_score']
        result['W_p_value']= with_watermark_detection_result['p_value']
        result['W_prediction']= with_watermark_detection_result['prediction']

        # Convert the result dictionary to a DataFrame
        df_result = pd.DataFrame([result])

        # Append the result DataFrame to the main DataFrame
        df = pd.concat([df, df_result], ignore_index=True)

        # Save the DataFrame as a CSV file
        df.to_csv('result.csv', index=False)


import pandas as pd


def main(args):
    args.normalizers = (args.normalizers.split(",") if args.normalizers else [])
    print(args)

    model, tokenizer, device = load_model(args)
    dataset = prepare_dataset(dataset_name='xsum', limit=8)

    # Read the existing results from 'result.csv' if it exists
    try:
        df_existing = pd.read_csv('result.csv')
    except FileNotFoundError:
        df_existing = pd.DataFrame()

    df = pd.DataFrame()

    if 'id' not in df_existing.columns:
        df_existing['id'] = None

    for input_text, id in tqdm(zip(dataset['document'], dataset['id'])):
        # Check if the ID already exists in the existing DataFrame
        if id in df_existing['id'].values:
            continue  # Skip processing this ID

        args.default_prompt = input_text
        _, _, decoded_output_without_watermark, decoded_output_with_watermark, _ = generate(input_text,
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

        result = {}
        result['id'] = id
        result['WO_num_tokens_scored'] = without_watermark_detection_result['num_tokens_scored']
        result['WO_num_green_tokens'] = without_watermark_detection_result['num_green_tokens']
        result['WO_green_fraction'] = without_watermark_detection_result['green_fraction']
        result['WO_z_score'] = without_watermark_detection_result['z_score']
        result['WO_p_value'] = without_watermark_detection_result['p_value']
        result['WO_prediction'] = without_watermark_detection_result['prediction']

        result['W_num_tokens_scored'] = with_watermark_detection_result['num_tokens_scored']
        result['W_num_green_tokens'] = with_watermark_detection_result['num_green_tokens']
        result['W_green_fraction'] = with_watermark_detection_result['green_fraction']
        result['W_z_score'] = with_watermark_detection_result['z_score']
        result['W_p_value'] = with_watermark_detection_result['p_value']
        result['W_prediction'] = with_watermark_detection_result['prediction']

        # Convert the result dictionary to a DataFrame
        df_result = pd.DataFrame([result])

        # Append the result DataFrame to the main DataFrame
        df = pd.concat([df, df_result], ignore_index=True)

        # Combine the existing DataFrame and the new results DataFrame
        df = pd.concat([df_existing, df], ignore_index=True)

        # Save the updated DataFrame as a CSV file
        df.to_csv('result.csv', index=False)

if __name__ == "__main__":
    # Load YAML file
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    args = Namespace()

    args.__dict__.update(config)

    main(args)
