from semora.text import remove_markdown_images


def test_remove_markdown_images_preserves_text_and_links() -> None:
    text = (
        "Before ![scan](images/scan.jpg \"page scan\") after.\n"
        "![](images/image-only.jpg)\n"
        "Keep [ordinary link](https://example.com)."
    )
    assert remove_markdown_images(text) == (
        "Before after.\n\nKeep [ordinary link](https://example.com)."
    )


def test_remove_markdown_images_handles_nested_parentheses_and_escaping() -> None:
    text = r"Text ![](images/file_(1).jpg) and \![](literal.jpg)."
    assert remove_markdown_images(text) == r"Text and \![](literal.jpg)."


def test_image_only_text_becomes_empty() -> None:
    assert remove_markdown_images("  ![](images/only.jpg)  ").strip() == ""
