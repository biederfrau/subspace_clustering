#!/usr/bin/env ruby
# encoding: utf-8

puts "\\begin{table}[]\\centering\n  \\begin{tabular}{lrrrcrrr}\\toprule\n"
puts "#{' '*4} & \\multicolumn{3}{c}{DM} & \\phantom{abc} & \\multicolumn{3}{c}{ELKI}\\\\"
puts "#{' '*4}dataset & min & max & avg & & min & max & avg\\\\ \\midrule"

files = ['test', 'test_noisy', 'higher_dimensional', 'paper']
files.each do |f|
  our_nmis = File.read("results/#{f}_NMI.txt").lines.map { |x| x.to_f }
  elki_nmis = Dir[File.join ['elki_results', "elki_#{f}*", 'cluster-evaluation.txt']].map { |f| File.read(f).lines.keep_if { |l| l.start_with? 'NMI Sqrt' }.map { |l| l.split.last.to_f } }.flatten

  avg = (our_nmis.reduce(&:+) / our_nmis.size).round 3
  avg_elki = (elki_nmis.reduce(&:+) / elki_nmis.size).round 3
  puts "#{' '*4}#{f.sub('_', ' ')} & #{our_nmis.min.round 3} & #{our_nmis.max.round 3} & #{avg > avg_elki ? '\textbf{' : avg == avg_elki ? '\emph{' : ''}#{avg}#{avg >= avg_elki ? '}' : ''} & & #{elki_nmis.min.round 3} & #{elki_nmis.max.round 3} & #{avg_elki}\\\\"

  File.open("#{f}_summary.dat", 'w+') do |fh|
    our_nmis.zip(elki_nmis).each do |o, e|
      fh.puts "#{o} #{e}"
    end
  end
end

puts "  \\bottomrule\n"
puts "  \\end{tabular}\n\\end{table}\n"

File.open('boxplt', 'w+') do |f|
  f.write "set style fill solid 0.25 border -1\n"
  f.write "set style fill solid 0.25 border -1\n"
  f.write "set style boxplot outliers pointtype 7\n"
  f.write "set style data boxplot\n"
  f.write "set boxwidth  0.15\n"
  f.write "set pointsize 0.5\n"
  f.write "set border 2\n"
  f.write "set xtics (\"test\" 1, \"test noisy\" 2, \"higher dimensional\" 3, \"paper\" 4) scale 0.0\n"
  f.write "set xtics nomirror\n"
  f.write "set ytics nomirror\n"
  f.write "set ylabel 'NMI'\n"
  f.write "set xlabel 'data set'\n"
  f.write "set terminal pdf\n"
  f.write "set output 'boxplt.pdf'\n"
  f.write "set key right bottom\n\n"

  f.write "set title 'NMI distributions' noenhanced\n"

  f.write <<~EOS
    plot 'test_summary.dat' using (1):1 lc 'red' title 'ours','' using (1.2):2 lt 2 title 'ELKI', \\
         'test_noisy_summary.dat' using (2):1 lc 'red' notitle, '' using (2.2):2 lt 2 notitle, \\
         'higher_dimensional_summary.dat' using (3):1 lc 'red' notitle, '' using (3.2):2 lt 2 notitle, \\
         'paper_summary.dat' using (4):1 lc 'red' notitle, '' using (4.2):2 lt 2 notitle'
  EOS
end

