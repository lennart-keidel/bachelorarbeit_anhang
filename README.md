# README

**Dies ist der Anhang zur Bachelorarbeit.**

**Alternativer Download Link:** [https://github.com/lennart-keidel/bachelorarbeit_anhang](https://github.com/lennart-keidel/bachelorarbeit_anhang)

**Titel:** Maschinelles Lernen in Kartenspielen mit imperfekten Informationen

**Bachelorarbeit-Nummer:**  BA AI 25/2023

**Autor:** Lennart Keidel

## Installation

Der Quellcode setzt die Installation von OpenSpiel voraus.

Genaue Installationsanweisungen sind in Abschnitt 4.2 der Bachelorarbeit zu finden.

Die Datei `requirements.txt` enthält alle benötigten Python-Packages, die nicht bereits durch OpenSpiel installiert werden.

## Ordner Struktur

`logs/` - Ordner für automatisch erstellte Logs während des Trainings

`src/` - die Python-Skripte

`src/training/` - die Python-Skripte zum trainieren der Modelle

`trained_models` - die trainierten Modelle, die in der Arbeit vorgestellt wurden

`train_all_models.sh` - Bash-Skript, führt alle Python Skripte zum trainieren der Modelle aus

## OpenSpiel

Der Quellcode basiert teilweise auf dem Quellcode von [OpenSpiel](https://github.com/deepmind/open_spiel).

Die OpenSpiel Lizenz ist in `license_OpenSpiel.md` enthalten.

Hier die Bibtex-Referenz zum OpenSpiel-Paper:

```bash
@article{LanctotEtAl2019OpenSpiel,
  title     = {{OpenSpiel}: A Framework for Reinforcement Learning in Games},
  author    = {Marc Lanctot and Edward Lockhart and Jean-Baptiste Lespiau and
               Vinicius Zambaldi and Satyaki Upadhyay and Julien P\'{e}rolat and
               Sriram Srinivasan and Finbarr Timbers and Karl Tuyls and
               Shayegan Omidshafiei and Daniel Hennes and Dustin Morrill and
               Paul Muller and Timo Ewalds and Ryan Faulkner and J\'{a}nos Kram\'{a}r
               and Bart De Vylder and Brennan Saeta and James Bradbury and David Ding
               and Sebastian Borgeaud and Matthew Lai and Julian Schrittwieser and
               Thomas Anthony and Edward Hughes and Ivo Danihelka and Jonah Ryan-Davis},
  year      = {2019},
  eprint    = {1908.09453},
  archivePrefix = {arXiv},
  primaryClass = {cs.LG},
  journal   = {CoRR},
  volume    = {abs/1908.09453},
  url       = {http://arxiv.org/abs/1908.09453},
}
```