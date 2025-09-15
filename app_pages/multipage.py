import streamlit as st


class MultiPage:
    def __init__(self, app_name):
        self.pages = []
        self.app_name = app_name

    def add_page(self, title, func):
        self.pages.append({"title": title, "function": func})

    def run(self):
        st.sidebar.title(self.app_name)  # Move app title to sidebar
        page_titles = [page["title"] for page in self.pages]
        selected_title = st.sidebar.radio("Menu", page_titles)

        # Find the page dict that matches selection
        for page in self.pages:
            if page["title"] == selected_title:
                page["function"]()  # Run the page
                break
