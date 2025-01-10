import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import boto3
from mypy_boto3_textract.type_defs import BlockTypeDef


# Function to read a file and convert its content into a bytearray
def get_document_data(file_name: str) -> bytearray:
    """
    Reads a file and converts its content to a bytearray.

    Args:
        file_name (str): The path to the file to be read.

    Returns:
        bytearray: The byte content of the file.
    """
    with open(file_name, "rb") as file:
        img = file.read()
        doc_bytes = bytearray(img)
        print(f"Image carregada: {file_name}")  # Logs the name of the loaded image
    return doc_bytes


# Function to analyze a document using AWS Textract and save the response to a JSON file
def analyze_document() -> None:
    """
    Calls AWS Textract's `analyze_document` API to process a document and save the response as a JSON file.

    Returns:
        None
    """
    client = boto3.client("textract")

    # Define the file path for the input image
    file_path = Path(__file__).parent / "images" / "cnh.png"

    # Read the file and convert it to bytearray
    doc_bytes = get_document_data(file_path)

    # Send the document for analysis with the feature type 'FORMS'
    response = client.analyze_document(
        Document={"Bytes": doc_bytes},
        FeatureTypes=["FORMS"],  # Specify the feature to detect
    )

    # Save the response from Textract to a JSON file
    with open("response.json", "w") as response_file:
        response_file.write(json.dumps(response))


# Function to extract key-value relationships, values, and blocks from the JSON response
def get_kv_map() -> Tuple[Dict[str, Dict], Dict[str, Dict], Dict[str, Dict]]:
    """
    Parses the JSON response from Textract to build mappings for keys, values, and blocks.

    Returns:
        Tuple[Dict, Dict, Dict]: A tuple containing:
            - key_map: Maps block IDs to key blocks.
            - value_map: Maps block IDs to value blocks.
            - block_map: Maps block IDs to all blocks.
    """
    key_map: Dict[str, Dict] = {}
    value_map: Dict[str, Dict] = {}
    block_map: Dict[str, Dict] = {}
    blocks: List[BlockTypeDef] = []

    try:
        # Open the saved Textract response JSON file
        with open("response.json", "r") as file:
            blocks = json.loads(file.read())["Blocks"]
    except IOError:
        # If the response file does not exist, analyze the document and retry
        analyze_document()
        with open("response.json", "r") as file:
            blocks = json.loads(file.read())["Blocks"]

    # Iterate over all blocks and categorize them into key, value, or block maps
    for block in blocks:
        block_id = block["Id"]
        block_map[block_id] = block

        if block["BlockType"] == "KEY_VALUE_SET":
            if "KEY" in block["EntityTypes"]:
                key_map[block_id] = block
            else:
                value_map[block_id] = block

    return key_map, value_map, block_map


# Function to extract relationships between keys and values
def get_kv_relationship(
    key_map: Dict[str, Dict], value_map: Dict[str, Dict], block_map: Dict[str, Dict]
) -> Dict:
    """
    Builds key-value relationships from the key, value, and block mappings.

    Args:
        key_map (Dict): Map of keys.
        value_map (Dict): Map of values.
        block_map (Dict): Map of all blocks.

    Returns:
        Dict: A dictionary of key-value pairs extracted from the document.
    """
    kvs = {}

    # Iterate over each key block
    for _, key_block in key_map.items():
        # Find the corresponding value block for the key
        value_block = find_value_block(key_block, value_map)
        key = get_text(key_block, block_map)  # Get the text for the key
        value = get_text(value_block, block_map) if value_block else ""  # Get the text for the value
        kvs[key] = value  # Map the key to the value

    return kvs


# Function to find the value block corresponding to a key block
def find_value_block(
    key_block: Dict[str, Dict], value_map: Dict[str, Dict]
) -> Optional[Dict]:
    """
    Finds the value block related to a given key block.

    Args:
        key_block (Dict): The key block to search for relationships.
        value_map (Dict): The map of value blocks.

    Returns:
        Optional[Dict]: The related value block, or None if no relationship is found.
    """
    for relationship in key_block.get("Relationships", []):
        if relationship["Type"] == "VALUE":
            for value_id in relationship["Ids"]:
                return value_map.get(value_id)
    return None


# Function to extract text from a block
def get_text(result: Dict[str, Dict], block_map: Dict[str, Dict]) -> str:
    """
    Extracts the text from a given block by traversing its child relationships.

    Args:
        result (Dict): The block to extract text from.
        block_map (Dict): The map of all blocks.

    Returns:
        str: The extracted text.
    """
    text = ""

    # Check if the block has child relationships
    if "Relationships" in result:
        for relationship in result["Relationships"]:
            if relationship["Type"] == "CHILD":
                for child_id in relationship["Ids"]:
                    word = block_map[child_id]
                    if word["BlockType"] == "WORD":
                        text += word["Text"] + " "  # Append the word text

    return text.rstrip()


# Main script execution
if __name__ == "__main__":
    """
    Main entry point for the script. Extracts key-value pairs from a document
    and prints them to the console.
    """
    # Retrieve the mappings for keys, values, and blocks
    key_map, value_map, block_map = get_kv_map()

    # Build key-value relationships from the mappings
    kvs = get_kv_relationship(key_map, value_map, block_map)

    # Print the extracted key-value pairs
    print("\n\n== DADOS DA CNH ==\n\n")
    for k, v in kvs.items():
        print(f"{k}: {v}")
