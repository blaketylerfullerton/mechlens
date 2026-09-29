"""Feature atlases built from this machine's own trained dictionaries.

The service validates a request and records it (`store`, `api`); the build
itself runs in a child process (`build`) so umap never enters the service and
cancelling is simply stopping that process.
"""
