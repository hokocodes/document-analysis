import os
from typing import Dict, Any, List
import json
from pathlib import Path
import pytesseract
from pdf2image import convert_from_path
from openai import OpenAI
from PIL import Image
import logging
import pandas as pd

class DocumentAnalyzer:
    def __init__(self, api_key: str, output_dir: str = "output"):
        """
        Initialize the document analyzer with OpenAI API key and output directory.
        
        Args:
            api_key (str): OpenAI API key
            output_dir (str): Directory for storing output files
        """
        self.client = OpenAI(api_key=api_key)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """
        Extract text from PDF using OCR.
        
        Args:
            pdf_path (str): Path to the PDF file
            
        Returns:
            str: Extracted text from the PDF
        """
        try:
            # Convert PDF to images
            images = convert_from_path(pdf_path)
            
            # Extract text from each image
            extracted_text = []
            for i, image in enumerate(images):
                self.logger.info(f"Processing page {i+1}")
                
                # Improve image quality for better OCR
                image = self._preprocess_image(image)
                
                # Extract text using Tesseract
                text = pytesseract.image_to_string(image)
                extracted_text.append(text)
            
            return "\n".join(extracted_text)
        
        except Exception as e:
            self.logger.error(f"Error in OCR processing: {str(e)}")
            raise

    def _preprocess_image(self, image: Image) -> Image:
        """
        Preprocess image to improve OCR accuracy.
        
        Args:
            image (PIL.Image): Input image
            
        Returns:
            PIL.Image: Processed image
        """
        # Convert to grayscale
        image = image.convert('L')
        
        # You can add more preprocessing steps here:
        # - Noise reduction
        # - Contrast enhancement
        # - Deskewing
        # - Thresholding
        
        return image

    def analyze_text_with_gpt(self, text: str) -> Dict[str, Any]:
        """
        Analyze extracted text using GPT-4 to identify key information.
        
        Args:
            text (str): Extracted text from document
            
        Returns:
            Dict[str, Any]: Structured data extracted from text
        """
        try:
            prompt = f"""
            Please analyze the following document text and extract key information in JSON format.
            Include the following fields where applicable:
            - dates
            - names (people and companies)
            - amounts/prices
            - key terms/conditions
            - document type
            
            Document text:
            {text}
            
            Provide the response in valid JSON format only.
            """
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",  # Updated model name
                messages=[
                    {"role": "system", "content": "You are a document analysis assistant. Extract structured information from documents and return it in JSON format."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Parse the response into JSON
            extracted_data = json.loads(response.choices[0].message.content)
            return extracted_data
            
        except Exception as e:
            self.logger.error(f"Error in GPT analysis: {str(e)}")
            raise

    def process_document(self, pdf_path: str, output_format: str = "json") -> str:
        """
        Process a document end-to-end: OCR -> GPT analysis -> structured output.
        
        Args:
            pdf_path (str): Path to the PDF file
            output_format (str): Desired output format ("json" or "csv")
            
        Returns:
            str: Path to the output file
        """
        # Extract text using OCR
        self.logger.info("Starting OCR processing...")
        extracted_text = self.extract_text_from_pdf(pdf_path)
        
        # Analyze text with GPT
        self.logger.info("Analyzing text with GPT...")
        analyzed_data = self.analyze_text_with_gpt(extracted_text)
        
        # Generate output filename
        input_filename = Path(pdf_path).stem
        output_filename = self.output_dir / f"{input_filename}_analyzed.{output_format}"
        
        # Save results in desired format
        if output_format == "json":
            with open(output_filename, 'w') as f:
                json.dump(analyzed_data, f, indent=2)
        elif output_format == "csv":
            df = pd.json_normalize(analyzed_data)
            df.to_csv(output_filename, index=False)
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
            
        self.logger.info(f"Analysis complete. Results saved to {output_filename}")
        return str(output_filename)

def main():
    # Example usage
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Please set the OPENAI_API_KEY environment variable")
    
    # Initialize analyzer
    analyzer = DocumentAnalyzer(api_key)
    
    # Process a single document
    pdf_path = r"C:\Users\hokop\Downloads\Texas Workforce Commission's Unemployment Tax Services - Texas Workforce Commission's Unemployment Tax Services _br__ Employer's Quarterly Report - Filed on April 30, 2024.pdf"
    output_file = analyzer.process_document(pdf_path, output_format="json")
    print(f"Processing complete. Results saved to: {output_file}")

if __name__ == "__main__":
    main()