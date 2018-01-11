from pkgutil import extend_path
__path__ = extend_path(__path__, __name__)  # type: ignore

__version__ = "0.3.2"
__repository__ = "https://github.com/usc-isi-i2/dsbox-cleaning"
__package_uri__ = "git+" + __repository__

__d3m_api_version__ = '2018.1.5'
__d3m_performer_team__ = 'ISI'

__installation_type__ = 'GIT'
if __installation_type__ == 'PYPI':
    __installation__ = {
        "type" : "PIP",
        "package": "dsbox-datacleaning",
        "version": __version__
    }
else:
    # __installation_type__ == 'GIT'
    __installation__ = {
        "type" : "PIP",
        "package_uri": __package_uri__,
    }
