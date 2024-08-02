import requests
from bs4 import BeautifulSoup
import os
import time

def get_script_urls(base_url="https://imsdb.com/all-scripts.html"):
    """
    Fetches all script URLs from the IMSDB all scripts page.
    """
    try:
        print(f"Starting to fetch script URLs from: {base_url}")
        response = requests.get(base_url, timeout=10)
        print(f"Received response from {base_url} with status code: {response.status_code}")
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            links = soup.find_all('a', href=True)
            script_urls = []
            for link in links:
                href = link['href']
                if href.startswith('/scripts/'):
                    script_url = "https://imsdb.com" + href
                    script_urls.append(script_url)
                    print(f"Found script URL: {script_url}")
            print(f"Total script URLs found: {len(script_urls)}")
            return script_urls
        else:
            print(f"Failed to fetch script URLs from {base_url}, status code: {response.status_code}")
            return []
    except requests.RequestException as e:
        print(f"Exception occurred while fetching script URLs: {e}")
        return []

def get_script_page_url(script_landing_url):
    """
    Fetches the actual script page URL from a script landing page.
    """
    try:
        print(f"Fetching script page URL from landing page: {script_landing_url}")
        response = requests.get(script_landing_url, timeout=10)
        print(f"Received response from {script_landing_url} with status code: {response.status_code}")
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            # Assumes that the script title can be extracted correctly from the page title
            script_title = soup.title.string.split(" Script")[0]
            print(f"Identified script title: {script_title}")
            # Finding the link with the text pattern matching "Read 'Movie Name' Script"
            link = soup.find('a', href=True, text=f'Read "{script_title}" Script')
            if link:
                script_page_url = "https://imsdb.com" + link['href']
                print(f"Found actual script page URL: {script_page_url}")
                return script_page_url
            else:
                print(f"No script page link found on landing page: {script_landing_url}")
                return None
        else:
            print(f"Failed to fetch script page URL from {script_landing_url}, status code: {response.status_code}")
            return None
    except requests.RequestException as e:
        print(f"Error fetching {script_landing_url}: {e}")
        return None

def scrape_script(script_url):
    """
    Extracts the script text from the script page.
    """
    try:
        print(f"Scraping script from URL: {script_url}")
        response = requests.get(script_url, timeout=10)
        print(f"Received response from {script_url} with status code: {response.status_code}")
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            script_text = soup.find('pre')
            if script_text:
                print(f"Successfully extracted script text from: {script_url}")
                return script_text.get_text(strip=True)
            else:
                print(f"Failed to find script content on page: {script_url}")
                return None
        else:
            print(f"Failed to access script page: {script_url}, status code: {response.status_code}")
            return None
    except requests.RequestException as e:
        print(f"Error scraping script from {script_url}: {e}")
        return None

def save_scripts(script_urls, folder_path='data/scripts', limit=10):
    """
    Saves a limited number of scripts from the provided URLs.
    """
    print(f"Starting to save scripts. Saving to folder: {folder_path}")
    os.makedirs(folder_path, exist_ok=True)
    for i, landing_url in enumerate(script_urls[:limit]):
        print(f"\nProcessing script {i+1}/{limit}")
        print(f"Fetching script landing page: {landing_url}")
        
        script_page_url = get_script_page_url(landing_url)
        if script_page_url:
            print(f"Fetching script content from script page: {script_page_url}")
            script = scrape_script(script_page_url)
            if script:
                file_path = os.path.join(folder_path, f'script_{i+1}.txt')
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(script)
                print(f"Saved script {i+1} to {file_path}")
            else:
                print(f"Failed to retrieve script content from {script_page_url}")
        else:
            print(f"Failed to find script page URL from landing page: {landing_url}")
        
        # Pause to avoid hitting rate limits
        print(f"Pausing for 1 second before next request...")
        time.sleep(1)

# Example usage:
# script_urls = get_script_urls()
# save_scripts(script_urls, limit=10)
