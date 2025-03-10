from tree_sitter import Language, Parser

Language.build_library(
  # Store the library in the `build` directory
  'build/my-languages.so',

  # Include one or more languages
  [
    './languages/tree-sitter-go',
    './languages/tree-sitter-javascript',
    './languages/tree-sitter-python',
    './languages/tree-sitter-php',
    './languages/tree-sitter-java',
    './languages/tree-sitter-c',
    './languages/tree-sitter-cpp',
    './languages/tree-sitter-c-sharp',
    './languages/tree-sitter-ruby'
  ]
)