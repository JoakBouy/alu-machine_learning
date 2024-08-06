#!/usr/bin/env python3
"""
this module contains the function count_logs"""

from pymongo import MongoClient

from pymongo import MongoClient


def check_logs():
    """
    Count and return logs
    """
    # Connect to MongoDB
    client = MongoClient()
    db = client.logs
    collection = db.nginx

    # Count total number of documents
    total_logs = collection.count_documents({})

    # Count number of documents with each method
    methods = ["GET", "POST", "PUT", "PATCH", "DELETE"]
    method_counts = {}
    for method in methods:
        count = collection.count_documents({"method": method})
        method_counts[method] = count

    # Count number of documents with method=GET and path=/status
    status_check_count = collection.count_documents(
        {"method": "GET", "path": "/status"})

    return total_logs, method_counts, status_check_count


if __name__ == "__main__":
    total_logs, method_counts, status_check_count = check_logs()

    print(f"{total_logs} logs")
    print("Methods:")
    for method, count in method_counts.items():
        print(f"\tmethod {method}: {count}")
    print(f"{status_check_count} status check")