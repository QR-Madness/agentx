import iso639
import torch
from termcolor import colored
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM

nlb200_list = ["ace_Arab", "ace_Latn", "acm_Arab", "acq_Arab", "aeb_Arab", "afr_Latn", "ajp_Arab", "aka_Latn",
               "amh_Ethi", "apc_Arab", "arb_Arab", "ars_Arab", "ary_Arab", "arz_Arab", "asm_Beng", "ast_Latn",
               "awa_Deva", "ayr_Latn", "azb_Arab", "azj_Latn", "bak_Cyrl", "bam_Latn", "ban_Latn", "bel_Cyrl",
               "bem_Latn", "ben_Beng", "bho_Deva", "bjn_Arab", "bjn_Latn", "bod_Tibt", "bos_Latn", "bug_Latn",
               "bul_Cyrl", "cat_Latn", "ceb_Latn", "ces_Latn", "cjk_Latn", "ckb_Arab", "crh_Latn", "cym_Latn",
               "dan_Latn", "deu_Latn", "dik_Latn", "dyu_Latn", "dzo_Tibt", "ell_Grek", "eng_Latn", "epo_Latn",
               "est_Latn", "eus_Latn", "ewe_Latn", "fao_Latn", "pes_Arab", "fij_Latn", "fin_Latn", "fon_Latn",
               "fra_Latn", "fur_Latn", "fuv_Latn", "gla_Latn", "gle_Latn", "glg_Latn", "grn_Latn", "guj_Gujr",
               "hat_Latn", "hau_Latn", "heb_Hebr", "hin_Deva", "hne_Deva", "hrv_Latn", "hun_Latn", "hye_Armn",
               "ibo_Latn", "ilo_Latn", "ind_Latn", "isl_Latn", "ita_Latn", "jav_Latn", "jpn_Jpan", "kab_Latn",
               "kac_Latn", "kam_Latn", "kan_Knda", "kas_Arab", "kas_Deva", "kat_Geor", "knc_Arab", "knc_Latn",
               "kaz_Cyrl", "kbp_Latn", "kea_Latn", "khm_Khmr", "kik_Latn", "kin_Latn", "kir_Cyrl", "kmb_Latn",
               "kon_Latn", "kor_Hang", "kmr_Latn", "lao_Laoo", "lvs_Latn", "lij_Latn", "lim_Latn", "lin_Latn",
               "lit_Latn", "lmo_Latn", "ltg_Latn", "ltz_Latn", "lua_Latn", "lug_Latn", "luo_Latn", "lus_Latn",
               "mag_Deva", "mai_Deva", "mal_Mlym", "mar_Deva", "min_Latn", "mkd_Cyrl", "plt_Latn", "mlt_Latn",
               "mni_Beng", "khk_Cyrl", "mos_Latn", "mri_Latn", "zsm_Latn", "mya_Mymr", "nld_Latn", "nno_Latn",
               "nob_Latn", "npi_Deva", "nso_Latn", "nus_Latn", "nya_Latn", "oci_Latn", "gaz_Latn", "ory_Orya",
               "pag_Latn", "pan_Guru", "pap_Latn", "pol_Latn", "por_Latn", "prs_Arab", "pbt_Arab", "quy_Latn",
               "ron_Latn", "run_Latn", "rus_Cyrl", "sag_Latn", "san_Deva", "sat_Beng", "scn_Latn", "shn_Mymr",
               "sin_Sinh", "slk_Latn", "slv_Latn", "smo_Latn", "sna_Latn", "snd_Arab", "som_Latn", "sot_Latn",
               "spa_Latn", "als_Latn", "srd_Latn", "srp_Cyrl", "ssw_Latn", "sun_Latn", "swe_Latn", "swh_Latn",
               "szl_Latn", "tam_Taml", "tat_Cyrl", "tel_Telu", "tgk_Cyrl", "tgl_Latn", "tha_Thai", "tir_Ethi",
               "taq_Latn", "taq_Tfng", "tpi_Latn", "tsn_Latn", "tso_Latn", "tuk_Latn", "tum_Latn", "tur_Latn",
               "twi_Latn", "tzm_Tfng", "uig_Arab", "ukr_Cyrl", "umb_Latn", "urd_Arab", "uzn_Latn", "vec_Latn",
               "vie_Latn", "war_Latn", "wol_Latn", "xho_Latn", "ydd_Hebr", "yor_Latn", "yue_Hant", "zho_Hans",
               "zho_Hant", "zul_Latn"]


class LanguageLexicon:
    """ Contains a list of languages and their corresponding language codes."""

    """ Contains a list of languages that are suitable for level I detection/translation. """
    level_i_languages = {"arabic": "ar", "bulgarian": "bg", "german": "de", "modern greek": "el", "english": "en",
                         "spanish": "es", "french": "fr", "hindi": "hi", "italian": "it", "japanese": "ja",
                         "dutch": "nl", "polish": "pl", "portuguese": "pt", "russian": "ru", "swahili": "sw",
                         "thai": "th", "turkish": "tr", "urdu": "ur", "vietnamese": "vi", "chinese": "zh"}

    """ Contains a list of languages that are suitable for level II translation. 
    Each entry is a terminological ISO639 code."""
    level_ii_languages = nlb200_list
    nlb200_list_mapped_to_iso639_terminological: dict

    def __init__(self, verbose=False):
        # construct the parsed lists
        self.nlb200_list_mapped_to_iso639_terminological = {language.split('_')[0]: language for language in
                                                            nlb200_list}

        if verbose:
            print("Initialized language lexicon.")
            print(colored("Level I languages", 'green'), ": ", colored(self.level_i_languages, "blue"))
            print(colored("Level II languages", 'green'), ": ", colored(self.level_ii_languages, "blue"))
            print(colored("Level I-II conversion list", 'green'), ": ",
                  colored(self.nlb200_list_mapped_to_iso639_terminological, "blue"))

    def convert_level_i_detection_to_level_ii(self, level_i_detected_language_code: str):
        language = None
        try:
            language = iso639.Language.from_part2t(level_i_detected_language_code.lower())
        except iso639.LanguageNotFoundError:
            try:
                language = iso639.Language.from_part1(level_i_detected_language_code.lower())
            except iso639.LanguageNotFoundError:
                raise ValueError(f"Language {level_i_detected_language_code} not supported for conversion to level II.")

        terminological_language_code = language.part2t
        if terminological_language_code not in self.nlb200_list_mapped_to_iso639_terminological:
            raise ValueError(f"Language {level_i_detected_language_code} (ISO639 {terminological_language_code}) is not supported by NLLB-200.")
        return self.nlb200_list_mapped_to_iso639_terminological[terminological_language_code]


class TranslationKit:
    """Translation models for language detection and translation."""
    language_lexicon = LanguageLexicon()

    language_detection_model_name = "eleldar/language-detection"
    level_i_translation_model_name = "facebook/nllb-200-distilled-600M"
    level_ii_translation_model_name = "facebook/nllb-200-distilled-600M"
    #

    def __init__(self):
        # initialize models
        self.language_detection_tokenizer = AutoTokenizer.from_pretrained(self.language_detection_model_name)
        self.language_detection_model = AutoModelForSequenceClassification.from_pretrained(
            self.language_detection_model_name)

        # Use NLLB-200 model for translation (supports 204 languages with NLLB format codes)
        self.translation_tokenizer = AutoTokenizer.from_pretrained(self.level_ii_translation_model_name)
        self.translation_model = AutoModelForSeq2SeqLM.from_pretrained(self.level_ii_translation_model_name)

    def detect_language_level_i(self, unclassified_text) -> tuple[str, float]:
        """ Will detect a language from a given text; supports about 20 languages; thus level I.
        Returns a tuple of (language, confidence).
        :param str unclassified_text: Text to detect language from.
        """
        inputs = self.language_detection_tokenizer(unclassified_text, return_tensors="pt")
        outputs = self.language_detection_model(**inputs)

        # Get predicted class
        predictions = torch.softmax(outputs.logits, dim=1)
        predicted_label_id = torch.argmax(predictions, dim=1).item()
        confidence = predictions[0][predicted_label_id].item()

        # Use the model's internal config to get the correct label
        detected_language = self.language_detection_model.config.id2label[predicted_label_id]

        return detected_language, round(confidence * 100, 2)

    def translate_text(self, text: str, target_language: str, target_language_level: int) -> str:
        """
        Translate text to a specific language using the translation model.
        Supports over 200 languages.
        Level I is the only supported option right now, due to usage of a distilled model, not suitable for dense text but lightweight.
        :param target_language: Language to translate to (format it based on parameter target_language_level).
        :param target_language_level: 1/Level 1 (format: "en"), 2/Level 2 (format: "eng_Latn").
        :param text: Text to translate.
        :return:
        """
        if target_language_level == 1:
            target_language_code = self.language_lexicon.convert_level_i_detection_to_level_ii(target_language)
        elif target_language_level == 2:
            if target_language not in self.language_lexicon.level_ii_languages:
                raise ValueError(f"Language {target_language} not supported for translation.")
            else:
                target_language_code = target_language
        else:
            raise ValueError(f"Invalid target language level: {target_language_level}")

        # convert language code to language name
        text_to_translate = text
        model_inputs = self.translation_tokenizer(text_to_translate, return_tensors="pt")

        # For NLLB-200, set the target language using forced_bos_token_id
        # Language codes are stored directly in the tokenizer vocabulary
        # self.translation_tokenizer.src_lang = "eng_Latn"  # Source language (default to English)

        # Get the token ID for the target language code from vocabulary
        forced_bos_token_id = self.translation_tokenizer.convert_tokens_to_ids(target_language_code)

        gen_tokens = self.translation_model.generate(
            **model_inputs,
            forced_bos_token_id=forced_bos_token_id,
            max_length=512
        )
        return self.translation_tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)[0]
