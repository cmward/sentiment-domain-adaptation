import re

class Tokenizer(object):
    """
    Tokenizes and preprocesses the sample's text, returning a
    list of tokens.

    Largely adapted from Christopher Potts' sentiment tokenizer:
    http://sentiment.christopherpotts.net/code-data/happyfuntokenizing.py

    url pattern adapted from http://stackoverflow.com/a/3809435/5818736
    """
    def __init__(self):
        self.emoticon_pattern = r'[=:;]-?\s?[\)\(D]'
        self.repeated_pattern = re.compile(r'(.)\1{2,}')
        self.user_pattern = re.compile(r'@+[\w_]+')
        self.hashtag_pattern = re.compile(r'\#+[\w_]+[\w\'_\-]*[\w_]+')
        self.url_pattern = re.compile(
            r'(https?:\/\/(www\.)?)?[-a-zA-Z0-9@:%._\+~#=]{2,256}\.' + \
            r'[a-z]{2,6}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)')
        word_pattern = r"""
            (?:<[^>]+>)                    # HTML tags
            |
            (?:[+\-]?\d+[,/.:-]\d+[+\-]?)  # Numbers, including fractions, decimals.
            |
            (?:[\w_\-']+)                  # Words
            |
            (?:\.(?:\s*\.){1,})            # Ellipsis dots.
            |
            (?:\S)                         # Everything else that isn't whitespace.
            """
        self.word_pattern = re.compile(word_pattern,
                                       re.VERBOSE | re.I | re.U)

    def tokenize(self, text):
        """
        Parameters
        ----------
        text: string to tokenize

        Returns
        -------
        tokens: list of tokens
        """
        # remove newlines
        try:
            text = unicode(text)
        except UnicodeDecodeError:
            text = str(text).encode('string_escape')
            text = unicode(text)
        # strip slashes
        text = re.sub(r'\\', '', text)
        # strip newlines
        text = re.sub(r'\\n', '', text)
        # map usernames to USERNAME
        text = re.sub(self.user_pattern, 'USER', text)
        # map hashtags to HASHTAG
        text = re.sub(self.hashtag_pattern, 'HASHTAG', text)
        # remove emoticons
        text = re.sub(self.emoticon_pattern, '', text)
        # map sequences of length >= 3 of the same character
        # to sequences of length 3 of the character
        text = re.sub(self.repeated_pattern, r'\g<1>\g<1>\g<1>', text)
        # map urls to URL
        text = re.sub(self.url_pattern, 'URL', text)
        tokens = self.word_pattern.findall(text)
        return tokens
