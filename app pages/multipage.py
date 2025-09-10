import streamlit as st

class MultiPage:
    """
    Class to manage multiple Streamlit pages in an object-oriented way.
    """

    def __init__(self, app_name: str) -> None:
        self.pages = []
        self.app_name = app_name

        # Configure Streamlit page
        st.set_page_config(
            page_title=self.app_name,
            page_icon="❤️",  # Heart icon for our heart disease project
            layout="wide",
        )

    def add_page(self, title: str, func) -> None:
        """
        Add a new page to the app.

        Parameters:
        - title: str -> page title displayed in sidebar
        - func: callable -> function that renders the page
        """
        self.pages.append({"title": title, "function": func})

    def run(self) -> None:
        """
        Run the multipage app with a sidebar for page selection.
        """
        st.title(self.app_name)
        page = st.sidebar.radio(
            "Navigation",
            self.pages,
            format_func=lambda page: page["title"]
        )
        page["function"]()
