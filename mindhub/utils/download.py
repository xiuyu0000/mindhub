"""Utility of downloading"""
import bz2
import gzip
import hashlib
import logging
import os
import ssl
import tarfile
import urllib
import urllib.error
import urllib.request
import requests
import zipfile
from copy import deepcopy
from typing import Optional, List

from tqdm import tqdm

from mindhub.env import GITHUB_REPO_URL
from mindhub.utils.path import detect_file_type

__all__ = [
    "get_default_download_root",
    "set_default_download_root",
    "DownLoad",
]

_logger = logging.getLogger(__name__)
# The default root directory where we save downloaded files.
# Use Get/Set to R/W this variable.
_DEFAULT_DOWNLOAD_ROOT = os.path.join(os.path.expanduser("~"), ".mindhub")


def get_default_download_root():
    return deepcopy(_DEFAULT_DOWNLOAD_ROOT)


def set_default_download_root(path):
    global _DEFAULT_DOWNLOAD_ROOT
    _DEFAULT_DOWNLOAD_ROOT = path


class DownLoad:
    """Base utility class for downloading."""

    HEADER: dict = {"User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/92.0.4515.131 Safari/537.36"
    )}

    @staticmethod
    def calculate_md5(file_path: str, chunk_size: int = 1024 * 1024) -> str:
        """Calculate md5 value."""
        md5 = hashlib.md5()
        with open(file_path, "rb") as fp:
            for chunk in iter(lambda: fp.read(chunk_size), b""):
                md5.update(chunk)
        return md5.hexdigest()

    def check_md5(self, file_path: str, md5: Optional[str] = None) -> bool:
        """Check md5 value."""
        return md5 == self.calculate_md5(file_path)

    @staticmethod
    def extract_tar(from_path: str, to_path: Optional[str] = None, compression: Optional[str] = None) -> None:
        """Extract tar format file."""

        with tarfile.open(from_path, f"r:{compression[1:]}" if compression else "r") as tar:
            tar.extractall(to_path)

    @staticmethod
    def extract_zip(from_path: str, to_path: Optional[str] = None, compression: Optional[str] = None) -> None:
        """Extract zip format file."""

        compression_mode = zipfile.ZIP_BZIP2 if compression else zipfile.ZIP_STORED
        with zipfile.ZipFile(from_path, "r", compression=compression_mode) as zip_file:
            zip_file.extractall(to_path)

    def extract_archive(self, from_path: str, to_path: str = None) -> str:
        """Extract and  archive from path to path."""
        archive_extractors = {
            ".tar": self.extract_tar,
            ".zip": self.extract_zip,
        }
        compress_file_open = {
            ".bz2": bz2.open,
            ".gz": gzip.open,
        }

        if not to_path:
            to_path = os.path.dirname(from_path)

        suffix, archive_type, compression = detect_file_type(from_path)  # pylint: disable=unused-variable

        if not archive_type:
            to_path = from_path.replace(suffix, "")
            compress = compress_file_open[compression]
            with compress(from_path, "rb") as rf, open(to_path, "wb") as wf:
                wf.write(rf.read())
            return to_path

        extractor = archive_extractors[archive_type]
        extractor(from_path, to_path, compression)

        return to_path

    def download_file(self, url: str, file_path: str, chunk_size: int = 1024):
        """Download a file."""
        # Define request headers.

        _logger.info(f"Downloading from {url} to {file_path} ...")
        with open(file_path, "wb") as f:
            request = urllib.request.Request(url, headers=self.HEADER)
            with urllib.request.urlopen(request) as response:
                with tqdm(total=response.length, unit="B") as pbar:
                    for chunk in iter(lambda: response.read(chunk_size), b""):
                        if not chunk:
                            break
                        pbar.update(chunk_size)
                        f.write(chunk)

    def download_url(
            self,
            url: str,
            path: Optional[str] = None,
            filename: Optional[str] = None,
            md5: Optional[str] = None,
            ) -> None:
        """Download a file from a url and place it in root."""
        if path is None:
            path = get_default_download_root()
        path = os.path.expanduser(path)
        os.makedirs(path, exist_ok=True)

        if not filename:
            filename = os.path.basename(url)

        file_path = os.path.join(path, filename)

        # Check if the file is exists.
        if os.path.isfile(file_path):
            if not md5 or self.check_md5(file_path, md5):
                return

        # Download the file.
        try:
            self.download_file(url, file_path)
        except (urllib.error.URLError, IOError) as e:
            if url.startswith("https"):
                url = url.replace("https", "http")
                try:
                    self.download_file(url, file_path)
                except (urllib.error.URLError, IOError):
                    # pylint: disable=protected-access
                    ssl._create_default_https_context = ssl._create_unverified_context
                    self.download_file(url, file_path)
                    ssl._create_default_https_context = ssl.create_default_context
            else:
                raise e

    def download_and_extract_archive(
            self,
            url: str,
            download_path: Optional[str] = None,
            extract_path: Optional[str] = None,
            filename: Optional[str] = None,
            md5: Optional[str] = None,
            remove_finished: bool = False,
    ) -> str:
        """Download and extract archive."""
        if download_path is None:
            download_path = get_default_download_root()
        download_path = os.path.expanduser(download_path)

        if not filename:
            filename = os.path.basename(url)

        self.download_url(url, download_path, filename, md5)

        archive = os.path.join(download_path, filename)
        self.extract_archive(archive, extract_path)

        if remove_finished:
            os.remove(archive)

        return download_path

    def list_remote_files(self, repo_url: str) -> List:
        response = requests.get(repo_url, headers=self.HEADER)

        if response.status_code == 404:
            raise ValueError(f"repository {repo_url} is not vaild")

        if response.status_code != 200:
            raise ValueError(f"Failed to get file list from repository {repo_url}")

        return response.json()

    def download_github_folder(self,
                               folder_path: str,
                               download_path: Optional[str] = None,
                               ) -> str:
        """
        Download the code of model.

        Args:
            folder_path(str): The model name.
            download_path(Optional[str]): The path to store the download file. Default: None.

        Returns:
            None
        """
        if download_path is None:
            download_path = get_default_download_root()
        download_path = os.path.join(os.path.expanduser(download_path), folder_path)

        # 解析仓库地址和文件夹路径
        repo_url = urllib.parse.urljoin(GITHUB_REPO_URL, folder_path)

        # 获取文件列表
        file_list = self.list_remote_files(repo_url)

        # 下载文件
        os.makedirs(download_path, exist_ok=True)
        for file_info in file_list:
            if file_info["type"] == "file":
                file_url = file_info["download_url"]
                self.download_url(file_url, download_path)
            elif file_info["type"] == "dir":
                subfolder_path = os.path.join(folder_path, file_info["name"])
                subfolder_download_path = os.path.join(download_path, file_info["name"])
                os.makedirs(subfolder_download_path, exist_ok=True)
                self.download_github_folder(subfolder_path, subfolder_download_path)

        return download_path


if __name__ == "__main__":
    download = DownLoad()
    download.download_github_folder("tinydarknet_imagenet")
