from importlib.metadata import metadata

meta = metadata("scikit-kda")

project = meta["Name"]
author = meta["Author-email"] or meta["Author"]
copyright = f"2018, {author}"
extensions = ["sphinx.ext.autodoc", "sphinx.ext.autosummary", "sphinx.ext.napoleon", "sphinx.ext.viewcode"]
autosummary_generate = True
exclude_patterns = ["_build"]
html_theme = "pydata_sphinx_theme"
autodoc_default_options = {"members": True, "show-inheritance": True}
