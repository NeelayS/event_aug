## Google Summer of Code (GSoC) 2022

### Project Details

- **Contributor:** Neelay Shah
- **Mentors**: Dr. James Turner and Prof. Thomas Nowotny
- **Project Title**: Creating Benchmark Datasets for Object Recognition with Event-based Cameras
- **Project-related links**:
  - [Code Repository](https://github.com/NeelayS/gsoc-2022/tree/gh-pages)
  - [GSoC Project Page](https://summerofcode.withgoogle.com/programs/2022/projects/dSlJsb1g)
  - [Blog](https://neelays.github.io/gsoc-2022/)

### Project Abstract

Event-based vision is a subfield of computer vision that deals with data from event-based cameras. Event cameras, also known as neuromorphic cameras, are bio-inspired imaging sensors that work differently to traditional cameras in that they measure pixel-wise brightness changes asynchronously instead of capturing images at a fixed rate. Since the way event cameras capture data is fundamentally different to traditional cameras, novel methods are required to process the output of these sensors. In addition to dealing with ways for capturing data with event cameras, event-based vision encompasses techniques to process the captured data - events - as well, including learning-based techniques and models, spiking neural networks (SNNs) being an example. This project aims to create benchmark datasets for object recognition tasks with event-based cameras. Using machine learning solutions for such tasks requires a sufficiently large and varied collection of data. The primary goal of this project is to develop Python utilities for augmenting event camera recordings of objects captured in an academic setting in various ways to create benchmark datasets. A secondary goal of this project is to test the performance of spiking neural networks for object recognition on the created datasets.

### Progress Log

#### Pre-coding Period

- Met with mentors (virtually), got to know about each other, and discussed project goals
- Set up GitHub code repository for the project
- Received data relevant to the project from mentors along with some starter code

#### Week 0 ( 6th June - 12th June)

- Added code for generating 2D Perlin noise
- Added code for spike encoding of image / video data using rate coding
- Set up continuous integration (CI) for the package which includes unit tests and linting and formatting checks

#### Week 1 ( 13th June - 19th June)

- Added code coverage check to CI workflow
- Created a GitHub pages site for the project
- Added code for spike encoding of video data using a method based on thresholding differences in intensities of pixels in neighbouring frames
- Added code to generate streams of progressive Perlin noise
- Worked on creating an automatic changelog generation workflow using GitHub Actions

<!-- ## Welcome to GitHub Pages

You can use the [editor on GitHub](https://github.com/NeelayS/gsoc-2022/edit/gh-pages/index.md) to maintain and preview the content for your website in Markdown files.

Whenever you commit to this repository, GitHub Pages will run [Jekyll](https://jekyllrb.com/) to rebuild the pages in your site, from the content in your Markdown files.

### Markdown

Markdown is a lightweight and easy-to-use syntax for styling your writing. It includes conventions for

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [Basic writing and formatting syntax](https://docs.github.com/en/github/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/NeelayS/gsoc-2022/settings/pages). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://docs.github.com/categories/github-pages-basics/) or [contact support](https://support.github.com/contact) and we’ll help you sort it out.
 -->
