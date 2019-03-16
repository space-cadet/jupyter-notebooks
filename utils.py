### Mathjax Custom Macros

# $ \newcommand{\opexpect}[3]{\langle #1 \vert #2 \vert #3 \rangle} $
# $ \newcommand{\rarrow}{\rightarrow} $
# $ \newcommand{\bra}{\langle} $
# $ \newcommand{\ket}{\rangle} $

# $ \newcommand{\up}{\uparrow} $
# $ \newcommand{\down}{\downarrow} $

# $ \newcommand{\mb}[1]{\mathbf{#1}} $
# $ \newcommand{\mc}[1]{\mathcal{#1}} $
# $ \newcommand{\mbb}[1]{\mathbb{#1}} $
# $ \newcommand{\mf}[1]{\mathfrak{#1}} $

# $ \newcommand{\vect}[1]{\boldsymbol{\mathrm{#1}}} $
# $ \newcommand{\expect}[1]{\langle #1\rangle} $

# $ \newcommand{\innerp}[2]{\langle #1 \vert #2 \rangle} $
# $ \newcommand{\fullbra}[1]{\langle #1 \vert} $
# $ \newcommand{\fullket}[1]{\vert #1 \rangle} $
# $ \newcommand{\supersc}[1]{^{\text{#1}}} $
# $ \newcommand{\subsc}[1]{_{\text{#1}}} $
# $ \newcommand{\sltwoc}{SL(2,\mathbb{C})} $
# $ \newcommand{\sltwoz}{SL(2,\mathbb{Z})} $

# $ \newcommand{\utilde}[1]{\underset{\sim}{#1}} $

# `switch` Class Definition

# Reference: [ActiveState Recipes](http://code.activestate.com/recipes/410692/)

# Example Usage:

# The following example is pretty much the exact use-case of a dictionary, but is included for its simplicity. Note that you can include statements in each suite.
    
# ```
# v = 'ten'
# for case in switch(v):
#     if case('one'):
#         print 1
#         break
#     if case('two'):
#         print 2
#         break
#     if case('ten'):
#         print 10
#         break
#     if case('eleven'):
#         print 11
#         break
#     if case(): # default, could also just omit condition or 'if True'
#         print "something else!"
#         # No need to break here, it'll stop anyway
# ```
# This class provides the functionality we want. You only need to look at
# this if you want to know how this works. It only needs to be defined
# once, no need to muck around with its internals.

class switch(object):
    def __init__(self, value):
        self.value = value
        self.fall = False

    def __iter__(self):
        """Return the match method once, then stop"""
        yield self.match
        raise StopIteration
    
    def match(self, *args):
        """Indicate whether or not to enter a case suite"""
        if self.fall or not args:
            return True
        elif self.value in args: # changed for v1.5, see below
            self.fall = True
            return True
        else:
            return False
        
# `hashQobj` Class Definition

# Provides a hashable version of Qobj(). In general to "hashify" any class one need only implement the `__hash__` method as shown below.

class hashQobj(Qobj):
    
    def __hash__(self):
        return hash(repr(self))

