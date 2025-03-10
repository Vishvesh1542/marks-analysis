from pypdf import PdfReader
import re
import time

# Easy to use this one function which does all the dirty work
# in getting the pdf so that the rest of the code can be
# clean
class PDFHandler:
    def __init__(self, pdf_path: str) -> None:
        if not pdf_path or not isinstance(pdf_path, str):
            raise ValueError("PDF path must be a non-empty string")
            
        self.pdf_path = pdf_path
        self.exam_type = "JEE"
    
    def cleanse_pdf(self):
        reader = PdfReader(self.pdf_path)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text

    def parse_pdf(self, text: str):
        pattern = r'(\d{6})\s+([A-Z\s]+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)'
        matches = re.findall(pattern, text)
        students_data = {}

        for match in matches:
            roll_number = match[0]
            name = match[1].strip()
            physics_marks = (int(match[2]), int(match[3]), int(match[4]), int(match[5]))  # Adjust according to the column position
            maths_marks = (int(match[6]), int(match[7]), int(match[8]), int(match[9]))    # Adjust according to the column position
            chemistry_marks = (int(match[10]), int(match[11]), int(match[12]), int(match[13]))  # Adjust according to the column position
            total_marks = (int(match[14]), int(match[15]), int(match[16]), int(match[17]))   # Adjust according to the column position
            students_data[roll_number] = {
                'name': name,
                'physics': physics_marks,
                'maths': maths_marks,
                'chemistry': chemistry_marks,
                'total': total_marks
            }
        return students_data


if __name__ == "__main__":
    t1 = time.time()

    pdf_reader = PDFHandler("test_3.pdf")
    raw_text = pdf_reader.cleanse_pdf()
    print(raw_text)
    parsed_data = pdf_reader.parse_pdf(raw_text)
    print(f"Time taken to process pdf: {time.time()-t1} seconds")
    print(parsed_data)