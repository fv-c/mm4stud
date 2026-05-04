\version "2.24.4"
\language "italiano"

\header {
  title = "Fra Martino"
  subtitle = "Nomi delle note in italiano"
  tagline = ##f
}

% Compilazione:
%   lilypond fra_martino.ly
%
% Il file produce sia la partitura sia il MIDI grazie al blocco \midi.

melodiaFraMartino = \relative do' {
  \key do \major
  \time 4/4
  \tempo 4 = 96

  do4 re mi do
  do4 re mi do
  mi4 fa sol2
  mi4 fa sol2
  sol8 la sol fa mi4 do
  sol8 la sol fa mi4 do
  re4 sol, do2
  re4 sol, do2
  \bar "|."
}

\score {
  <<
    \new NoteNames {
      \set printOctaveNames = ##f
      \melodiaFraMartino
    }
    \new Staff \with {
      instrumentName = "Canto"
      midiInstrument = "flute"
    } {
      \clef treble
      \melodiaFraMartino
    }
  >>
  \layout { }
  \midi { }
}
