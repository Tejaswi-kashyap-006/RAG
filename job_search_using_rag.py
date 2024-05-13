import logging
import tempfile
from datetime import datetime
import pandas as pd
from linkedin_jobs_scraper import LinkedinScraper
from linkedin_jobs_scraper.events import Events, EventData, EventMetrics
from linkedin_jobs_scraper.query import Query, QueryOptions, QueryFilters
from linkedin_jobs_scraper.filters import (
    RelevanceFilters,
    TimeFilters,
    TypeFilters,
)
from dotenv import load_dotenv
import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage
from llama_parse import LlamaParse
import nest_asyncio
import sys
import gradio as gr

load_dotenv()
nest_asyncio.apply()
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

PERSIST_DIR = "./storage"
job_postings = []

parser = LlamaParse(result_type="markdown")
file_extractor = {".pdf": parser}


def on_data(data: EventData):
    print(
        "[ON_DATA]",
        data.title,
        data.company,
        data.company_link,
        data.date,
        data.link,
        data.insights,
        len(data.description),
    )
    job_postings.append([data.job_id, data.location, data.title, data.company, data.date, data.link,
                         data.description, ])

    df = pd.DataFrame(job_postings, columns=['Job_ID', 'Location', 'Title', 'Company', 'Date', 'Link', 'Description'])
    df.to_csv(r"C:\Users\tejas\PycharmProjects\pocs\data\jobs.csv")


# Fired once for each page (25 jobs)
def on_metrics(metrics: EventMetrics):
    print("[ON_METRICS]", str(metrics))


def on_error(error):
    print("[ON_ERROR]", error)


def on_end():
    print("[ON_END]")


def initialise_scraper():
    scraper = LinkedinScraper(
        chrome_executable_path=None,  # Custom Chrome executable path (e.g. /foo/bar/bin/chromedriver)
        chrome_binary_location=None,
        # Custom path to Chrome/Chromium binary (e.g. /foo/bar/chrome-mac/Chromium.app/Contents/MacOS/Chromium)
        chrome_options=None,  # Custom Chrome options here
        headless=True,
        max_workers=1,
        slow_mo=0.5,
        page_load_timeout=40,
    )
    scraper.on(Events.DATA, on_data)
    scraper.on(Events.ERROR, lambda error: print("[ON_ERROR]", error))
    scraper.on(Events.END, lambda: print("[ON_END]"))
    return scraper


def scrape_jobs(job_title: str, locations: list):
    scraper = initialise_scraper()
    queries = [Query(query=job_title, options=QueryOptions(locations=locations,
                                                           filters=QueryFilters(relevance=RelevanceFilters.RECENT,
                                                                                time=TimeFilters.MONTH,
                                                                                type=[TypeFilters.FULL_TIME]),
                                                           limit=10))]
    scraper.run(queries)


def create_vector_storage():
    if not os.path.exists(PERSIST_DIR):
        documents = SimpleDirectoryReader("data").load_data()
        index = VectorStoreIndex.from_documents(documents)
        index.storage_context.persist(persist_dir=PERSIST_DIR)
    else:
        logging.error(f"{PERSIST_DIR} already exists")


def sanitize_filename(filename):
    """Create a safe filename by removing disallowed characters."""
    valid_chars = "-_.() %s%s" % (string.ascii_letters, string.digits)
    sanitized_filename = ''.join(c for c in filename if c in valid_chars)
    return sanitized_filename


def save_resume_to_folder(resume_file):
    resumes_folder = "resumes"
    os.makedirs(resumes_folder, exist_ok=True)
    base_name = os.path.basename(resume_file.name)
    sanitized_name = sanitize_filename(base_name)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"{timestamp}_{sanitized_name}"
    filepath = os.path.join(resumes_folder, filename)
    with open(filepath, "wb") as f:
        f.write(resume_file.read())
    return filepath


def user_query(question, pdf_path: str):
    input_cv = parser.load_data(pdf_path)

    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)

    query_engine = index.as_query_engine()
    response = query_engine.query(
        "You are a brilliant career adviser. Answer a question of job seekers with given information.\n"
        "If their CV information is given, use that information as well to answer the question.\n"
        "If you are asked to return jobs that are suitable for the job seeker, return Job ID, Title and Link.\n"
        "If you are not sure about the answer, return NA. \n"
        "You need to show the source nodes that you are using to answer the question at the end of your response.\n"
        f"CV: {input_cv[0]} \n"
        f"Question: {question}"
    )
    return response


last_inputs_cache = {"job_title": None, "locations": None}


def main(job_title, locations, resume_path, question):
    global last_inputs_cache
    # resume_path = save_resume_to_folder(resume_file)
    locations_list = [location.strip() for location in locations.split(',')]
    # Step 1: Scrape jobs with the given job title and locations
    inputs_changed = (job_title != last_inputs_cache["job_title"] or locations_list != last_inputs_cache["locations"])
    print("Scraping jobs...")
    if inputs_changed:
        print("Inputs changed, scraping jobs...")
        scrape_jobs(job_title, locations_list)
        print("Job scraping completed.")
    else:
        print("Inputs unchanged, skipping job scraping.")
    print("Creating/updating vector storage...")
    try:
        create_vector_storage()
        print("Vector storage is ready.")
    except Exception as e:
        return f"Error during vector storage creation: {str(e)}"

    # Step 3: Process the user's query with their uploaded CV to find suitable jobs
    print("Processing your query based on the uploaded CV...")
    try:
        response = user_query(question, resume_path)
        return response
    except Exception as e:
        return f"Error during query processing: {str(e)}"


iface = gr.Interface(
    fn=main,
    inputs=[
        gr.Textbox(label="Job Title"),
        gr.Textbox(label="Locations"),
        gr.File(label="Upload your CV", type="filepath"),
        gr.Textbox(label="Question")  # Added input for the question
    ],
    outputs="text",
    description="Enter the job title, locations (comma-separated), upload your CV, and your question to get personalized job recommendations."
)

if __name__ == "__main__":
    iface.launch(share=True)
