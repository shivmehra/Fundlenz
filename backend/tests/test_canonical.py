from app.id.canonical import AliasMap, normalize_name


def test_normalize_drops_noise_tokens():
    assert normalize_name("HDFC Top 100 Fund") == "hdfc top 100"
    assert normalize_name("The HDFC Scheme") == "hdfc"


def test_normalize_keeps_distinguishing_qualifiers():
    # Regular vs Direct vs Growth are not noise — they distinguish real schemes.
    assert normalize_name("HDFC Top 100 Regular Plan") == "hdfc top 100 regular"
    assert normalize_name("HDFC Top 100 Direct Growth") == "hdfc top 100 direct growth"
    assert normalize_name("HDFC Top 100 IDCW") == "hdfc top 100 idcw"


def test_normalize_is_idempotent():
    raw = "  Aditya Birla Sun Life - LARGE CAP Fund (Regular)  "
    once = normalize_name(raw)
    assert normalize_name(once) == once


def test_normalize_handles_unicode_and_punctuation():
    assert normalize_name("ICICI—Prudential / Bluechip") == "icici prudential bluechip"
    assert normalize_name("") == ""
    assert normalize_name(None) == ""  # type: ignore[arg-type]


def test_alias_map_register_and_resolve(tmp_path):
    am = AliasMap(tmp_path / "aliases.json")
    am.load()
    canon = am.register("HDFC Top 100 Fund")
    assert canon == "hdfc top 100"
    assert am.resolve("HDFC Top 100") == "hdfc top 100"
    assert am.resolve("hdfc top 100 fund") == "hdfc top 100"
    assert am.resolve("Some Other Fund") is None


def test_alias_map_roundtrip(tmp_path):
    p = tmp_path / "aliases.json"
    am1 = AliasMap(p)
    am1.load()
    am1.register("HDFC Top 100 Fund")
    am1.register("Axis Bluechip Fund")
    am1.save()

    am2 = AliasMap(p)
    am2.load()
    assert am2.resolve("HDFC Top 100") == "hdfc top 100"
    assert am2.resolve("Axis Bluechip") == "axis bluechip"
    assert "HDFC Top 100 Fund" in am2.aliases_for("hdfc top 100")


def test_alias_map_does_not_duplicate_aliases(tmp_path):
    am = AliasMap(tmp_path / "aliases.json")
    am.load()
    am.register("HDFC Top 100 Fund")
    am.register("HDFC Top 100 Fund")
    aliases = am.aliases_for("hdfc top 100")
    assert aliases.count("HDFC Top 100 Fund") == 1
