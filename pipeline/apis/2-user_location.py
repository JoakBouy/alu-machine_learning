#!/usr/bin/env python3
""" Script for getting user location"""
import sys
import requests
import time


def get_user_location(api_url):
    """
    Get the location of a user from the provided API URL.

    Args:
        api_url (str): The API URL of the user.

    Returns:
        str: The location of the user.
    """
    response = requests.get(api_url)
    if response.status_code == 200:
        user_data = response.json()
        if 'location' in user_data:
            return user_data['location']
        else:
            return "Location not available"
    elif response.status_code == 404:
        return "Not found"
    elif response.status_code == 403:
        reset_time = int(response.headers['X-Ratelimit-Reset'])
        current_time = time.time()
        reset_in_seconds = reset_time - current_time
        reset_in_minutes = int(reset_in_seconds / 60)
        return "Reset in {} min".format(reset_in_minutes)
    else:
        return "Unexpected error"


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python 2-user_location.py <API_URL>")
        sys.exit(1)

    api_url = sys.argv[1]
    location = get_user_location(api_url)
    print(location)