import concurrent
import io
import json
import os
import urllib
import urllib.parse
from concurrent.futures import ThreadPoolExecutor

import boto3
import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm.auto import tqdm


class CoralNetDownloader:
    """Main downloader class for CoralNet sources using requests"""

    CORALNET_URL = "https://coralnet.ucsd.edu"
    LOGIN_URL = "https://coralnet.ucsd.edu/accounts/login/"

    def __init__(self, username: str, password: str):
        self.username = username
        self.password = password
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
            }
        )
        self.logged_in = False
        self.s3 = boto3.client("s3")

    def login(self) -> bool:
        """Log in to CoralNet using requests session"""
        success = False
        try:
            # Get login page to extract CSRF token
            response = self.session.get(self.LOGIN_URL, timeout=30)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")
            csrf_token = soup.find("input", attrs={"name": "csrfmiddlewaretoken"})

            if not csrf_token:
                raise Exception("Could not find CSRF token")

            # Prepare login data
            data = {
                "username": self.username,
                "password": self.password,
                "csrfmiddlewaretoken": csrf_token["value"],
            }

            headers = {"Referer": self.LOGIN_URL}

            # Submit login
            login_response = self.session.post(
                self.LOGIN_URL, data=data, headers=headers, timeout=30, allow_redirects=True
            )

            # Check if login was successful by looking for sign out button or redirect
            if "Sign out" in login_response.text or login_response.url != self.LOGIN_URL:
                success = True
                self.logged_in = True
                print("✓ Login successful")
            else:
                raise Exception("Login failed - invalid credentials or other error")

        except Exception as e:
            print(f"ERROR: Could not login with {self.username}: {str(e)}")

        return success

    def check_permissions(self, source_id: int) -> bool:
        """Check permissions for accessing a source"""
        try:
            url = f"{self.CORALNET_URL}/source/{source_id}/"
            response = self.session.get(url, timeout=30)
            response.raise_for_status()

            if "Page could not be found" in response.text:
                raise Exception("Source does not exist")
            elif "don't have permission" in response.text:
                raise Exception("Permission denied")

            return True

        except Exception as e:
            print(f"ERROR: Permission check failed for source {source_id}: {str(e)}")
            return False

    def download_metadata(
        self, source_id: int, bucket_name: str, s3_prefix: str = "coralnet-public-images"
    ) -> tuple[bool, int]:
        """Download metadata for a source"""
        success = False
        total_images_number = 0

        try:
            url = f"{self.CORALNET_URL}/source/{source_id}/"
            response = self.session.get(url, timeout=30)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")

            # Try to get total images count
            try:
                image_status_header = soup.find("h4", string="Image Status")
                if image_status_header:
                    table = image_status_header.find_next_sibling(
                        "table", class_="detail_box_table"
                    )
                    if table:
                        # Find the row where one of the <td> contains 'Total images:'
                        total_images_row = None
                        for tr in table.find_all("tr"):
                            tds = tr.find_all("td")
                            if any("Total images:" in td.get_text() for td in tds):
                                total_images_row = tr
                                break
                        if total_images_row:
                            link = total_images_row.find("a")
                            if link:
                                try:
                                    total_images_number = int(
                                        link.get_text().strip().replace(",", "")
                                    )
                                except Exception:
                                    total_images_number = 0
                                print(f"Total images: {total_images_number}")
            except Exception as e:
                print(f"Warning: Can't get number of images: {e}")
                total_images_number = 0

            # Extract classifier plot data from JavaScript
            script_tags = soup.find_all("script")
            classifier_data = None

            for script in script_tags:
                if script.string and "Classifier overview" in script.string:
                    script_text = script.string
                    start_marker = "let classifierPlotData = "
                    start_index = script_text.find(start_marker)

                    if start_index != -1:
                        start_index += len(start_marker)
                        end_index = script_text.find("];", start_index) + 1
                        classifier_plot_data_str = script_text[start_index:end_index]

                        # Clean up JavaScript object notation to valid JSON
                        classifier_plot_data_str = classifier_plot_data_str.replace("'", '"')

                        try:
                            classifier_data = json.loads(classifier_plot_data_str)
                            break
                        except json.JSONDecodeError as e:
                            print(f"Warning: Could not parse classifier data: {e}")

            if not classifier_data:
                print("No metadata found for this source")
                return True, total_images_number

            # Process classifier data
            meta = []
            for point in classifier_data:
                meta.append(
                    [
                        point.get("x"),  # classifier_nbr
                        point.get("y"),  # score
                        point.get("nimages"),  # nimages
                        point.get("traintime"),  # traintime
                        point.get("date"),  # date
                        point.get("pk"),  # src_id
                    ]
                )

            # Save metadata
            meta_df = pd.DataFrame(
                meta,
                columns=[
                    "Classifier nbr",
                    "Accuracy",
                    "Trained on",
                    "Date",
                    "Traintime",
                    "Global id",
                ],
            )
            print(meta_df)

            # Save to S3 instead of local file
            csv_buffer = io.StringIO()
            meta_df.to_csv(csv_buffer, index=False)

            s3_key = f"{s3_prefix}/s{source_id}/metadata.csv"

            # Upload to S3
            self.s3.put_object(
                Bucket=bucket_name, Key=s3_key, Body=csv_buffer.getvalue(), ContentType="text/csv"
            )

            print(f"✓ Metadata saved to s3://{bucket_name}/{s3_key}")
            success = True

        except Exception as e:
            print(f"ERROR: Issue downloading metadata: {str(e)}")

        return success, total_images_number

    def download_labelset(
        self, source_id: int, bucket_name: str, s3_prefix: str = "coralnet-public-images"
    ) -> bool:
        """Download labelset for a source"""
        success = False

        try:
            url = f"{self.CORALNET_URL}/source/{source_id}/labelset/"
            response = self.session.get(url, timeout=30)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")
            table = soup.find("table", {"id": "label-table"})

            if table is None:
                raise Exception("Unable to find the label table")

            rows = table.find_all("tr")
            if not rows or len(rows) <= 1:  # Only header row or no rows
                print("No labelset found for this source")
                return True

            label_ids = []
            names = []
            short_codes = []

            for row in rows[1:]:  # Skip header row
                cells = row.find_all("td")
                if cells:
                    # Get label ID from link
                    link = cells[0].find("a")
                    if link and link.get("href"):
                        label_id = link["href"].split("/")[-2]
                        label_ids.append(label_id)

                        # Get name
                        names.append(link.get_text().strip())

                        # Get short code (second column)
                        if len(cells) > 1:
                            short_codes.append(cells[1].get_text().strip())
                        else:
                            short_codes.append("")

            if label_ids:
                labelset_df = pd.DataFrame(
                    {"Label ID": label_ids, "Name": names, "Short Code": short_codes}
                )

                # Save to S3 instead of local file
                csv_buffer = io.StringIO()
                labelset_df.to_csv(csv_buffer, index=False)

                s3_key = f"{s3_prefix}/s{source_id}/labelset.csv"

                # Upload to S3
                self.s3.put_object(
                    Bucket=bucket_name,
                    Key=s3_key,
                    Body=csv_buffer.getvalue(),
                    ContentType="text/csv",
                )
                # filepath = os.path.join(output_dir, "labelset.csv")
                # labelset_df.to_csv(filepath, index=False)

                print(f"✓ Labelset saved to s3://{bucket_name}/{s3_key}")
                success = True

            else:
                print("No labels found in labelset")
                success = True

        except Exception as e:
            print(f"ERROR: Issue downloading labelset: {str(e)}")

        return success

    def download_annotations(
        self, source_id: int, bucket_name: str, s3_prefix: str = "coralnet-public-images"
    ) -> bool:
        """Download annotations for a source"""
        success = False

        try:
            # First, get the browse images page to extract form data
            browse_url = f"{self.CORALNET_URL}/source/{source_id}/browse/images/"
            response = self.session.get(browse_url, timeout=30)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")

            # Find the export form
            export_form = soup.find("form", {"id": "export-annotations-prep-form"})
            if not export_form:
                raise Exception("Could not find export annotations form")

            # Extract CSRF token
            csrf_token = export_form.find("input", {"name": "csrfmiddlewaretoken"})
            if not csrf_token:
                raise Exception("Could not find CSRF token in export form")

            # Prepare form data for annotation export
            form_data = {
                "csrfmiddlewaretoken": csrf_token["value"],
                "browse_action": "export_annotations",
                "image_select_type": "all",
                "label_format": "both",
                # Add all optional columns
                "optional_columns": [
                    "annotator_info",
                    "metadata_date_aux",
                    "metadata_other",
                ],
            }

            export_request_url = f"{self.CORALNET_URL}/source/{source_id}/annotation/export_prep/"
            # Submit the export request
            export_response = self.session.post(
                export_request_url,
                headers={"Referer": browse_url},
                data=form_data,
                timeout=300,  # Longer timeout for processing
                allow_redirects=True,
            )

            export_timestamp = export_response.json()["session_data_timestamp"]
            download_annotations_url = f"https://coralnet.ucsd.edu/source/{source_id}/export/serve/?session_data_timestamp={export_timestamp}"
            download_annotations_response = self.session.get(download_annotations_url, timeout=60)
            download_annotations_response.raise_for_status()

            df_annotations = pd.read_csv(io.StringIO(download_annotations_response.text))
            # Save to S3 instead of local file
            csv_buffer = io.StringIO()
            df_annotations.to_csv(csv_buffer, index=False)

            s3_key = f"{s3_prefix}/s{source_id}/annotations.csv"

            # Upload to S3
            self.s3.put_object(
                Bucket=bucket_name, Key=s3_key, Body=csv_buffer.getvalue(), ContentType="text/csv"
            )
            # filepath = os.path.join(output_dir, "labelset.csv")
            # labelset_df.to_csv(filepath, index=False)

            print(f"✓ Annotations saved to s3://{bucket_name}/{s3_key}")
            success = True
            # annotations_file = os.path.join(output_dir, "annotations.csv")
            # if not os.path.exists(output_dir):
            #     os.makedirs(output_dir)
            # df_annotations.to_csv(annotations_file, index=False)
            # if os.path.exists(annotations_file) and os.path.getsize(annotations_file) > 0:
            #     print(f"✓ Annotations saved to {annotations_file}")
            #     success = True
            # else:
            #     raise Exception("Downloaded annotations file is empty")

        except Exception as e:
            print(f"ERROR: Issue downloading annotations: {str(e)}")
            success = False  # Don't fail the entire process

        return success

    def get_images_on_page(self, browse_url: str) -> tuple[dict[str, str], str | None]:
        """
        Get a dictionary of image names and their URLs from the CoralNet browse page

        Args:
            session: requests.Session object with valid CoralNet login
            browse_url: URL of the browse images page

        Returns:
            dict: Dictionary with image names as keys and their URLs as values
        """
        images = {}

        response = self.session.get(browse_url, timeout=30)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        thumb_wrappers = soup.find_all("span", class_="thumb_wrapper")
        for wrapper in thumb_wrappers:
            link = wrapper.find("a")
            img = wrapper.find("img")
            if link and img:
                image_name = img.get("title", "")
                image_url = link.get("href", "")

                if image_name and image_url:
                    images[image_name] = image_url

        next_page_element = soup.find("a", title="Next page")
        next_page_url = next_page_element.get("href") if next_page_element else None

        return images, next_page_url

    def get_images(self, source_id: int) -> tuple[pd.DataFrame | None, bool]:
        """
        Get a DataFrame of all images from a CoralNet source

        Args:
            session: requests.Session object with valid CoralNet login
            source_id: ID of the CoralNet source

        Returns:
            pd.DataFrame: DataFrame containing image names and URLs
        """
        images = None
        success = False

        base_url = f"{self.CORALNET_URL}/source/{source_id}/browse/images"
        all_images = {}
        try:
            imgs, next_page = self.get_images_on_page(base_url)
            all_images.update(imgs)
            p_bar = tqdm(desc="Fetching images", unit="page")
            while next_page:
                imgs, next_page = self.get_images_on_page(f"{base_url}/{next_page}")
                all_images.update(imgs)
                p_bar.update(1)
            p_bar.close()
            success = True
            images = pd.DataFrame(list(all_images.items()), columns=["Name", "Image Page"])
        except Exception as e:
            print(f"ERROR: Issue retrieving images: {str(e)}")
        return images, success

    def get_image_urls(self, image_page_urls: list[str]) -> list[str | None]:
        """
        Get the direct image URLs from the CoralNet image page URLs

        Args:
            image_page_url: URL of the CoralNet image page

        Returns:
            str or None: Direct image URL or None if not found
        """
        image_urls = []
        for image_page_url in tqdm(image_page_urls, desc="Fetching image URLs", unit="image"):
            image_page_url = f"https://coralnet.ucsd.edu{image_page_url}"
            image_view_response = urllib.request.urlopen(image_page_url)
            response_soup = BeautifulSoup(image_view_response.read(), "html.parser")

            original_img_elements = response_soup.select("div#original_image_container > img")
            if not original_img_elements:
                raise ValueError(
                    f"CoralNet image {image_page_url}: couldn't find image on the"
                    f" image-view page. Maybe it's in a private source."
                )
            image_url = original_img_elements[0].attrs.get("src")
            image_urls.append(image_url)

        return image_urls

    @staticmethod
    def download_image(url: str, path: str, timeout: int = 30) -> tuple[str, bool]:
        """Download a single image"""
        if os.path.exists(path):
            return path, True

        try:
            response = requests.get(url, timeout=timeout, stream=True)
            response.raise_for_status()

            os.makedirs(os.path.dirname(path), exist_ok=True)

            with open(path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

            if os.path.exists(path) and os.path.getsize(path) > 0:
                return path, True
            else:
                return path, False

        except Exception as e:
            print(f"Warning: Failed to download {url}: {e}")
            return path, False

    def download_image_to_s3(
        self, url: str, bucket_name: str, s3_key: str, timeout: int = 30
    ) -> tuple[str, bool]:
        """Download a single image and upload directly to S3"""
        try:
            # Check if object already exists in S3
            try:
                self.s3.head_object(Bucket=bucket_name, Key=s3_key)
                return s3_key, True  # File already exists
            except self.s3.exceptions.ClientError:
                pass  # File doesn't exist, continue with download

            response = requests.get(url, timeout=timeout, stream=True)
            response.raise_for_status()

            # Upload directly to S3 without saving locally
            self.s3.upload_fileobj(
                response.raw, bucket_name, s3_key, ExtraArgs={"ContentType": "image/jpeg"}
            )

            return s3_key, True

        except Exception as e:
            print(f"Warning: Failed to download {url} to S3: {e}")
            return s3_key, False

    def download_images(
        self,
        images_df: pd.DataFrame,
        source_id: int,
        bucket_name: str,
        s3_prefix: str = "coralnet-public-images",
    ) -> None:
        """Download all images from a DataFrame"""
        # Save image list
        csv_buffer = io.StringIO()
        images_df.to_csv(csv_buffer, index=False)

        s3_key = f"{s3_prefix}/s{source_id}/image_list.csv"

        # Upload to S3
        self.s3.put_object(
            Bucket=bucket_name, Key=s3_key, Body=csv_buffer.getvalue(), ContentType="text/csv"
        )

        print(f"✓ Images saved to s3://{bucket_name}/{s3_key}")
        success = True

        # Filter out rows without URLs
        valid_images = images_df[images_df["Image URL"].notna()]

        if valid_images.empty:
            print("Warning: No valid image URLs found")
            return

        print(f"Downloading {len(valid_images)} images...")

        with ThreadPoolExecutor(max_workers=min(8, os.cpu_count() or 4)) as executor:
            futures = []
            for _, row in valid_images.iterrows():
                name = (
                    row["Image Page"].replace("/image/", "").replace("/view/", "")
                )  # .split("image/")[1].split("/view/")[0] #row['Name']
                url = row["Image URL"]
                clean_name = (
                    name + ".jpg"
                )  # name.replace(".jpg", "").replace(" - Confirmed", "") + ".jpg"
                s3_key = f"{s3_prefix}/s{source_id}/images/{clean_name}"

                # path = os.path.join(image_dir.replace(".jpg", ""), name.replace(".jpg", "").replace(" - Confirmed", "") + ".jpg")
                # print(path)
                futures.append(executor.submit(self.download_image_to_s3, url, bucket_name, s3_key))

                # futures.append(executor.submit(self.download_image, url, path))

            completed = 0
            successful = 0
            for future in concurrent.futures.as_completed(futures):
                try:
                    path, success = future.result()
                    if success:
                        successful += 1
                    else:
                        print(f"Warning: Failed to download {os.path.basename(path)}")
                except Exception as e:
                    print(f"ERROR: {str(e)}")

                completed += 1
                if completed % 10 == 0 or completed == len(futures):
                    print(f"Progress: {completed}/{len(futures)} images processed")

        print(
            f"✓ Uploaded {successful}/{len(valid_images)} images to s3://{bucket_name}/{s3_prefix}/s{source_id}/images/"
        )

    def download_source(
        self,
        source_id: int,
        bucket_name: str,
        s3_prefix: str = "coralnet-public-images",
        download_metadata: bool = True,
        download_labelset: bool = True,
        download_annotations: bool = True,
        download_images: bool = True,
    ) -> bool:
        """Download all data for a source"""
        print(f"\n=== Downloading Source {source_id} ===")

        # Login if needed
        if not self.logged_in:
            if not self.login():
                raise Exception("Failed to login to CoralNet")

        # Check permissions
        if not self.check_permissions(source_id):
            print(f"Cannot access source {source_id}")
            return True
            # raise Exception(f"Cannot access source {source_id}")

        success = True
        n_images = 0

        # Download metadata
        if download_metadata:
            metadata_success, n_images = self.download_metadata(
                source_id, bucket_name=bucket_name, s3_prefix=s3_prefix
            )
            if not metadata_success:
                print("Warning: Failed to download metadata")

            if n_images == 0:
                print("Source appears to be empty")
                # os.makedirs(os.path.join(source_dir, "empty"), exist_ok=True)
                return True

        # Download labelset
        if download_labelset:
            if not self.download_labelset(source_id, bucket_name=bucket_name, s3_prefix=s3_prefix):
                print("Warning: Failed to download labelset")

        # Download annotations
        if download_annotations:
            if not self.download_annotations(
                source_id, bucket_name=bucket_name, s3_prefix=s3_prefix
            ):
                print("Warning: Failed to download annotations")
                return success  # Temporary addition to skip very large sources

        # Download images
        if download_images:
            images_df, images_success = self.get_images(source_id)
            if images_success and images_df is not None and len(images_df) > 0:
                # Get image URLs
                image_urls = self.get_image_urls(images_df["Image Page"].tolist())
                images_df["Image URL"] = image_urls

                # Download images
                self.download_images(
                    images_df, source_id, bucket_name=bucket_name, s3_prefix=s3_prefix
                )
            else:
                print("Warning: No images found or failed to retrieve image list")

        print(f"✓ Completed downloading source {source_id}")
        return success

    def cleanup(self):
        """Clean up resources"""
        if self.session:
            self.session.close()
        self.logged_in = False
        self.logged_in = False
