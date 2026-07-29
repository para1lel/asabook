export const pseudocodeLanguage = {
  name: 'pseudocode',
  aliases: ['pseudo'],
  scopeName: 'source.pseudocode',
  patterns: [
    { include: '#comments' },
    { include: '#directives' },
    { include: '#keywords' },
    { include: '#constants' },
    { include: '#functions' },
    { include: '#numbers' },
    { include: '#variables' },
    { include: '#operators' },
    { include: '#punctuation' },
  ],
  repository: {
    comments: {
      begin: '//',
      beginCaptures: {
        0: { name: 'punctuation.definition.comment.pseudocode' },
      },
      end: '$',
      name: 'comment.line.double-slash.pseudocode',
    },
    directives: {
      match: '^(Input|Output)(:)',
      captures: {
        1: { name: 'keyword.other.directive.pseudocode' },
        2: { name: 'punctuation.separator.pseudocode' },
      },
    },
    keywords: {
      match: '\\b(?:function|if|then|else|for|while|do|return|in|where|is|not|and|or|Add|Remove)\\b',
      name: 'keyword.control.pseudocode',
    },
    constants: {
      match: '\\b(?:true|false|null|empty)\\b',
      name: 'constant.language.pseudocode',
    },
    functions: {
      match: '\\b[A-Za-z_][A-Za-z0-9_]*(?=\\s*\\()',
      name: 'entity.name.function.pseudocode',
    },
    numbers: {
      match: '\\b(?:0[xX][0-9A-Fa-f]+|\\d+(?:\\.\\d+)?)\\b',
      name: 'constant.numeric.pseudocode',
    },
    variables: {
      match: "\\b(?:[A-Z](?:\\d+)?(?:_[A-Za-z0-9]+)*|[a-z]+(?:_[A-Za-z0-9]+)+|alpha|beta|gamma|delta|epsilon|zeta|eta|theta|iota|kappa|lambda|mu|nu|xi|omicron|pi|rho|sigma|tau|upsilon|phi|chi|psi|omega)'?(?![A-Za-z0-9_])",
      name: 'variable.other.pseudocode',
    },
    operators: {
      match: '<=|>=|!=|==|[=+*/<>-]',
      name: 'keyword.operator.pseudocode',
    },
    punctuation: {
      match: '[(){}\\[\\],.:]',
      name: 'punctuation.pseudocode',
    },
  },
}
