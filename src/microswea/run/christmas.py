#!/usr/bin/env python3
"""
Christmas tree script that prints a Christmas tree and exits.
"""

def print_christmas_tree():
    """Print a beautiful Christmas tree."""
    tree = """
        🌟
       /|\\
      /*|O\\
     /*/|\\*\\
    /X/O|*\\X\\
   /*/X/|\\X\\*\\
  /O/*/X|*\\O\\X\\
 /*/O/X/|\\X\\O\\*\\
/X/O/*/X|O\\X\\*\\O\\
        |X|
        |X|
"""
    print(tree)
    print("🎄 Merry Christmas! 🎄")

def main():
    """Main entry point for the Christmas tree script."""
    print_christmas_tree()

if __name__ == "__main__":
    main()
