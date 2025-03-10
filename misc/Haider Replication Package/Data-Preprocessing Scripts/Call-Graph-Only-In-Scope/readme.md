## Installation

This package currently only works with Python 3. There are no library dependencies, but you do need to have a C compiler installed.

```sh
pip3 install tree_sitter
```


#### Setup

First you'll need a Tree-sitter language implementation for each language that you want to parse. Put all these file into directory languages 

```sh
mkdir languages
cd languages
git clone https://github.com/tree-sitter/tree-sitter-c
git clone https://github.com/tree-sitter/tree-sitter-cpp
git clone https://github.com/tree-sitter/tree-sitter-c-sharp
git clone https://github.com/tree-sitter/tree-sitter-java
git clone https://github.com/tree-sitter/tree-sitter-php
git clone https://github.com/tree-sitter/tree-sitter-go
git clone https://github.com/tree-sitter/tree-sitter-ruby
git clone https://github.com/tree-sitter/tree-sitter-javascript
git clone https://github.com/tree-sitter/tree-sitter-php
```

#### Build

Build the tree-sitters using the python file

```sh
python3 ./build-languages.py
```

#### Run

Run call graph by passing language in commandline argument `python3 ./callGraph.py <language>`
```sh
python3 ./callGraph.py python
```

It will generate the callGraph for corresponding test file



