#!/usr/bin/perl -w
#
#

use strict;

($#ARGV >= 0) or die "usage $0 <qrels> <sysrels>+";

my ($file,$rank);
my %judgment;
my %topics;

my %interesting = (
		   1 => 1,
		   5 => 1,
		   10 => 1,
		   50 => 1,
		   100 => 1,
		   500 => 1,
		   1000 => 1
		  );

# READ IN TOPICS
$file = shift(@ARGV);
if ($file =~ /\.(z|gz)$/i) {
  open(FILE, "gunzip -c $file |") or die "Cannot open $file";
} else {
  open(FILE, $file) or die "Cannot open $file";
}
while (<FILE>) {
  chomp;
  my ($t,$d,$r) = (split)[0,2,3];
  $judgment{"$t:$d"} = $r;
  $topics{$t}++;
}
close(FILE);
my $num_top = scalar(keys %topics);

# Process Sysrels
while ($#ARGV >= 0) {
  $file = shift(@ARGV);
  if ($file =~ /\.(z|gz)$/i) {
    open(FILE, "gunzip -c $file |") or die "Cannot open $file";
  } else {
    open(FILE, $file) or die "Cannot open $file";
  }
  my %rel;
  my %nonrel;
  my %unjudged;
  my $old_topic = "-1";
  while (<FILE>) {
    chomp;
    next if (!/\w/);  # Skip empty lines.
    my ($t,$d) = (split)[0,2];
    if ($t ne $old_topic) {
      $rank = 0;
      $old_topic = $t
    }
    next if (!exists($topics{$t}));
    if (!exists($judgment{"$t:$d"})) {
      # Unjudged
      foreach (keys %interesting) {
	$unjudged{$_}++ if ($_ > $rank);
      }
    } elsif ($judgment{"$t:$d"}) {
      # Rel
      foreach (keys %interesting) {
	$rel{$_}++ if ($_ > $rank);
      }
    } else {
      # Nonrel
      foreach (keys %interesting) {
	$nonrel{$_}++ if ($_ > $rank);
      }
    }
    $rank++;
  }
  close(FILE);
  print "Results for $file over $num_top topics:\n";
  foreach (sort {$a <=> $b} keys %interesting) {
    $rel{$_} = 0 if (!exists $rel{$_});
    # print "JK: $_ $rel{$_}\n";
    printf "Relevant:    %4d %6d %6.2f (%6.2f%%)\n", 
      $_, $rel{$_}, ($rel{$_}/$num_top), (($rel{$_} * 100)/($_ * $num_top));
    $nonrel{$_} = 0 if (!exists $nonrel{$_});
    printf "Nonrelevant: %4d %6d %6.2f (%6.2f%%)\n", 
      $_, $nonrel{$_}, $nonrel{$_}/$num_top, (($nonrel{$_}/$_) * 100)/$num_top;
    $unjudged{$_} = 0 if (!exists $unjudged{$_});
    printf "Unjudged:    %4d %6d %6.2f (%6.2f%%)\n", 
      $_, $unjudged{$_}, $unjudged{$_}/$num_top, 
	(($unjudged{$_}/$_) * 100)/$num_top;
  }
}

exit;




