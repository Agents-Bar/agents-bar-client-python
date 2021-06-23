import logging
import os
from typing import Dict, Optional

import requests

USERNAME_KEY = "AGENTS_BAR_USER"
PASSWORD_KEY = "AGENTS_BAR_PASS"


class Client(object):
    """
    Session object that stores credentials and configuration.

    """

    default_url = "https://agents.bar"
    logger = logging.getLogger("AgentsBar")

    def __init__(self, username: Optional[str] = None, password: Optional[str] = None, base_url: Optional[str] = None):
        """
        Initiates session to Agents Bar. If credentials aren't passed directly then it expects them
        to be present in environment variables as `AGENTS_BAR_USER` and `AGENTS_BAR_PASS`.

        Parameters:
            username (optional str): Username required for login. Usually an email. Looks in env vars if None passed.
            password (optional str): Password associated with username. Looks in env vars if None passed.
            base_url (optional str): Service location. Defaults to `https://agents.bar`.

        """
        if username is None and password is None:
            # Look in env only if neither is passed
            username = os.environ[USERNAME_KEY]
            password = os.environ[PASSWORD_KEY]
        
        if username is None or password is None:
            raise ValueError("No credentials provided for logging in. Please pass either 'access_token' or "
                             "('username' and 'password'). These credentials should be related to your Agents Bar account.")
        
        self._username = username
        self._base_url: str = self.__parse_url(base_url)
        self.__access_token: str = self.__login(username, password)

        self._headers = {"Authorization": f"Bearer {self.__access_token}", "accept": "application/json"}

    @property
    def username(self):
        return self._username

    @staticmethod
    def __parse_url(base_url: Optional[str] = None) -> str:
        "Determins full API url based on provided (or not) `base_url`."
        if base_url is None:
            base_url = Client.default_url
        assert base_url.startswith("http"), "Base url needs to start with either `http` or `https`"
        assert base_url.split(":", 1)[0] in ("http", "https"), "Only http and https protocols are supported"
        assert base_url[-1] != "/", "Base url cannot end with `/`"

        return base_url + "/api/v1"

    
    def __login(self, username: str, password: str) -> str:

        data = dict(username=username, password=password)
        response = requests.post(f"{self._base_url}/login/access-token", data=data)

        if response.status_code >= 300:
            self.logger.error(response.text)
            raise ValueError(
                f"Received an error while trying to authenticate as username='{username}'. "
                f"Please double check your credentials. Error: {response.text}"
            )
        return response.json()['access_token']

    def get(self, url: str, params: Optional[Dict] = None):
        return requests.get(self._base_url + url, headers=self._headers, params=params)
    
    def post(self, url: str, data: Optional[Dict] = None, params: Optional[Dict] = None):
        return requests.post(self._base_url + url, json=data, headers=self._headers, params=params)
    
    def delete(self, url: str):
        return requests.delete(self._base_url + url, headers=self._headers)

    def put(self, url: str):
        return requests.put(self._base_url + url, headers=self._headers)
