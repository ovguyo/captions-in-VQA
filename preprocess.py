import os

def remove_newline(input_string):
    cleaned_string = input_string.replace("\n", "")
    return cleaned_string


def remove_punc(input_string):
    cleaned_string = input_string.replace(".", "")
    return cleaned_string


def remove_s(input_string):
    cleaned_string = input_string.replace("</s>", "")
    return cleaned_string


def remove_comma(input_string):
    cleaned_string = input_string.replace(",", "")
    return cleaned_string


def remove_the(input_string):
    if input_string.startswith("the "):
        input_string = input_string.replace("the ", "")
    return input_string


def remove_a(input_string):
    if input_string.startswith("a "):
        input_string = input_string.replace("a ", "")
    return input_string


def remove_an(input_string):
    if input_string.startswith("an "):
        input_string = input_string.replace("an ", "")
    return input_string


def get_cleaned(input_string):
    cleaned_string = remove_newline(input_string)
    cleaned_string = remove_punc(cleaned_string)
    cleaned_string = remove_comma(cleaned_string)
    cleaned_string = remove_s(cleaned_string)
    cleaned_string = remove_the(cleaned_string)
    cleaned_string = remove_a(cleaned_string)
    cleaned_string = remove_an(cleaned_string)
    return cleaned_string