css="""
    body {
        background-color: #141414 !important;
        color: white !important;
    }

    .container { 
        max-width: 90%; 
    }

    .title {
        color: fff !important;
        text-align: center;
        padding: 10px 0;
        background-color: #1f2121;
    }

    #output-container {
        max-height: 400px;
        overflow-y: auto;
        padding: 10px;
        border: 1px solid #ccc;
        border-radius: 5px;
    }

"""

js = """
function refresh() {
    const url = new URL(window.location);

    if (url.searchParams.get('__theme') !== 'dark') {
        url.searchParams.set('__theme', 'dark');
        window.location.href = url.href;
    }
}
"""