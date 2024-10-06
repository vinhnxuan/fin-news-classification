
<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
        <li><a href="#usage">Usage</a></li>
      </ul>
    </li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

[![Product Name Screen Shot][product-screenshot]](https://example.com)

Financial News Sentiment Analysis: this project aims to build lib for fine-tuning llama2 (or even LLM-GPT) and T2 (or even encoder-decoder LLM) for doing sentiment analysis classification task.

The preparation and pre-processing of dataset is written for a common CSV dataset. Their output can be used and fed into the model for an easy use

LSTM is used as baseline method.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running follow these simple example steps.
Docker version will be upated later

### Prerequisites

PYTHON 3 (>=3.9)
Jupyter Notebook
Streamlit

### Installation
1/ Step 1: Create enviroment (using Anaconda)
```
conda create -n test_fnsa python==3.9
```
2/ Step 2: install libraries
```
pip install -r requirements.txt
```
3/ Step 3: run notebook src/scripts/finetune_llama2.ipynb with the created env test_fnsa
4/ Step 4: run streamlit app
```
pip install streamlit
cd src/
streamlit run app.py --server.port 8080
```
<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- USAGE EXAMPLES -->
## Usage

Use this space to show useful examples of how a project can be used. Additional screenshots, code examples and demos work well in this space. You may also link to more resources.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- CONTACT -->
## Contact

Vinh Nguyen - xuanvinh1609@yahoo.com

Project Link: [https://github.com/vinhnxuan/fin-news-classification](https://github.com/vinhnxuan/fin-news-classification)

<p align="right">(<a href="#readme-top">back to top</a>)</p>
