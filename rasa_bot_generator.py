import json

def faq_domain_generator():
    """

    :return:
    """
    with open('test_bot/domain.yml', 'w', encoding='utf-8') as story_file:
        with open('labels.json', 'r', encoding='utf-8') as label_file:
            labels_as_intents = json.load(label_file)
            story_file.write('intents: \n')
            for intent in labels_as_intents:
                story_file.write(f'  - {intent}\n')
            story_file.write('actions: \n')
            for intent in labels_as_intents:
                story_file.write(f'  - utter_{intent}\n')

def faq_story_generator():
    """"""

    with open('test_bot/data/stories.md', 'w', encoding='utf-8') as story_file:
        with open('labels.json', 'r', encoding='utf-8') as label_file:
            labels_as_intents = json.load(label_file)
            for intent in labels_as_intents:
                story_file.write(f'## faq_story_{intent}\n')
                story_file.write(f'* {intent}\n')
                story_file.write(f'  - utter_{intent}\n')
                story_file.write(f'  - action_restart\n')


if __name__ == "__main__":
    faq_story_generator()
    faq_domain_generator()