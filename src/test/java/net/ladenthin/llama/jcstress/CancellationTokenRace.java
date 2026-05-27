// SPDX-FileCopyrightText: 2026 Bernard Ladenthin <bernard.ladenthin@gmail.com>
//
// SPDX-License-Identifier: MIT
package net.ladenthin.llama.jcstress;

import net.ladenthin.llama.CancellationToken;
import org.openjdk.jcstress.annotations.Actor;
import org.openjdk.jcstress.annotations.Arbiter;
import org.openjdk.jcstress.annotations.Description;
import org.openjdk.jcstress.annotations.Expect;
import org.openjdk.jcstress.annotations.JCStressTest;
import org.openjdk.jcstress.annotations.Outcome;
import org.openjdk.jcstress.annotations.State;
import org.openjdk.jcstress.infra.results.Z_Result;

@JCStressTest
@Description("cancel() must be visible to the arbiter via the volatile flag.")
@Outcome(id = "true",  expect = Expect.ACCEPTABLE, desc = "Cancellation visible after actor completes")
@Outcome(id = "false", expect = Expect.FORBIDDEN,  desc = "BUG: volatile write not seen after actor finish")
@State
public class CancellationTokenRace {

    private final CancellationToken token = new CancellationToken();

    @Actor
    public void writer() {
        token.cancel();
    }

    @Arbiter
    public void check(Z_Result r) {
        r.r1 = token.isCancelled();
    }
}
